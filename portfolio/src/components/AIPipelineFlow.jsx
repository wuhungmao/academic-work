import { useCallback, useState } from 'react'
import ReactFlow, {
  Background,
  Controls,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
} from 'reactflow'
import 'reactflow/dist/style.css'
import './AIPipelineFlow.css'

const NODE_INFO = {
  input:        { label: '🎬 Input Video',         desc: 'Raw video file fed into the pipeline. Both visual frames and audio track are extracted.' },
  face:         { label: '👤 Face Detection\n(MTCNN)',     desc: 'MTCNN locates and crops faces from each frame. Only face regions are passed to the visual models to reduce noise.' },
  audio:        { label: '🔊 Audio Extraction\n(Librosa)',  desc: 'Separates the audio track from the video for independent analysis of speech and audio artifacts.' },
  preprocess:   { label: '🖼 Video Preprocessing\n(Albumentations)', desc: 'Applies augmentations and normalization to cropped face frames before feeding them to the visual models.' },
  efficientnet: { label: '⚡ EfficientNet-B1',    desc: 'Fine-tuned on AIGVDBench and FaceForensics++. Detects diffusion and face-transfer deepfake artifacts in video frames.' },
  xceptionnet:  { label: '✖ XceptionNet',         desc: 'Pre-trained on FaceForensics++. Excels at detecting face-swap and neural-rendering artifacts using depthwise separable convolutions.' },
  mesonet:      { label: '🔍 MesoNet',             desc: 'Lightweight CNN targeting mesoscopic image properties. Optimized for face manipulation detection with low computational cost.' },
  aasist:       { label: '🎵 AASIST',              desc: 'Audio Anti-Spoofing using Integrated Spectro-Temporal graph attention. Detects synthesized or manipulated speech in the audio stream.' },
  ensemble:     { label: '⚖ Weighted Ensemble',    desc: 'Combines predictions from all 4 models using weighted averaging. Parallel inference is used so all models run concurrently.' },
  output:       { label: '🏁 REAL / FAKE',          desc: 'Final binary prediction with confidence score. Threshold can be tuned for precision vs recall trade-off.' },
}

function PipelineNode({ data }) {
  const [hovered, setHovered] = useState(false)
  return (
    <div
      className={`pipeline-node ${data.type}`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <Handle type="target" position={Position.Left} />
      <div className="pipeline-node-label">{data.label}</div>
      {hovered && (
        <div className="pipeline-node-tooltip">{data.desc}</div>
      )}
      <Handle type="source" position={Position.Right} />
    </div>
  )
}

const nodeTypes = { pipeline: PipelineNode }

function makeNode(id, x, y, type = 'default') {
  return {
    id,
    type: 'pipeline',
    position: { x, y },
    data: { label: NODE_INFO[id].label, desc: NODE_INFO[id].desc, type },
  }
}

const initialNodes = [
  makeNode('input',        0,   140, 'source'),
  makeNode('face',       180,    60, 'process'),
  makeNode('audio',      180,   220, 'process'),
  makeNode('preprocess', 380,    60, 'process'),
  makeNode('efficientnet',580,    0,  'model'),
  makeNode('xceptionnet', 580,   80, 'model'),
  makeNode('mesonet',     580,  160, 'model'),
  makeNode('aasist',      580,  240, 'model'),
  makeNode('ensemble',    800,  120, 'ensemble'),
  makeNode('output',      990,  120, 'output'),
]

const initialEdges = [
  { id: 'e1', source: 'input',        target: 'face',         animated: true, style: { stroke: '#3d444d' } },
  { id: 'e2', source: 'input',        target: 'audio',        animated: true, style: { stroke: '#3d444d' } },
  { id: 'e3', source: 'face',         target: 'preprocess',   animated: true, style: { stroke: '#3d444d' } },
  { id: 'e4', source: 'preprocess',   target: 'efficientnet', animated: true, style: { stroke: '#3d444d' } },
  { id: 'e5', source: 'preprocess',   target: 'xceptionnet',  animated: true, style: { stroke: '#3d444d' } },
  { id: 'e6', source: 'preprocess',   target: 'mesonet',      animated: true, style: { stroke: '#3d444d' } },
  { id: 'e7', source: 'audio',        target: 'aasist',       animated: true, style: { stroke: '#3d444d' } },
  { id: 'e8', source: 'efficientnet', target: 'ensemble',     animated: true, style: { stroke: '#26a641' } },
  { id: 'e9', source: 'xceptionnet',  target: 'ensemble',     animated: true, style: { stroke: '#26a641' } },
  { id: 'e10',source: 'mesonet',      target: 'ensemble',     animated: true, style: { stroke: '#26a641' } },
  { id: 'e11',source: 'aasist',       target: 'ensemble',     animated: true, style: { stroke: '#26a641' } },
  { id: 'e12',source: 'ensemble',     target: 'output',       animated: true, style: { stroke: '#f0f6fc', strokeWidth: 2 } },
]

export default function AIPipelineFlow() {
  const [nodes, , onNodesChange] = useNodesState(initialNodes)
  const [edges, , onEdgesChange] = useEdgesState(initialEdges)

  return (
    <div className="ai-flow-wrap">
      <div className="ai-flow-title">AI Deepfake Detection — Pipeline Architecture</div>
      <div className="ai-flow-hint">Hover over any node for details · Drag to pan · Scroll to zoom</div>
      <div className="ai-flow-canvas">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.4}
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#30363d" gap={20} size={1} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </div>
  )
}

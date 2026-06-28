const publications = [
  {
    id: 'ai-video-paper',
    title: 'Ensemble Deep Learning Approaches for AI-Altered Video Detection',
    authors: 'Wu, H.-M., Bi, F., Abdelhadi, Y., Lin, W., Khan, L.',
    venue: 'Computer Science Implementation Project (CSC492), University of Toronto Mississauga',
    date: 'Apr 2026',
    status: 'unpublished',
    description: 'Proposes a multimodal ensemble architecture combining EfficientNet-B1, MesoNet, XceptionNet, and AASIST to detect AI-generated and deepfake video. Details fine-tuning strategies on AIGVDBench and FaceForensics++, MTCNN-based face detection, Albumentations preprocessing, parallel inference with weighted scoring, and ensemble architectural design.',
    pdf: 'https://drive.google.com/file/d/1KaVo7DM9onu-ap9cOaXpPX9bg1vr4znx/view?usp=sharing',
    tags: ['Deep Learning', 'Deepfake Detection', 'Ensemble Methods', 'Computer Vision'],
  },
  {
    id: 'agents-debug-paper',
    title: 'Can Agents Debug Better Than a Software Engineer?',
    authors: 'Riku, T., Ahmed, K., Wu, H.-M., Usman, R.',
    venue: 'Course Research White Paper (CSC398), University of Toronto Mississauga',
    date: 'Apr 2026',
    status: 'unpublished',
    description: 'Evaluates the bug-localization and automated fix capabilities of LLM-based coding agents (Claude Code, Codex) against the JetBrains LCA benchmark. Extends the benchmark from file-level bug localization to automated patch generation, and introduces expanded evaluation metrics including token usage, iteration count, and fix success rate.',
    pdf: 'https://drive.google.com/file/d/1YPqEK_rhctJtD1Wj-wzSBSfd9r5xmhBI/view?usp=sharing',
    tags: ['LLM Agents', 'Software Engineering', 'Benchmarking', 'Claude Code', 'Codex'],
  },
]

export default publications

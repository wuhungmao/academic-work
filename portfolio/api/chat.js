import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { streamText, tool, convertToModelMessages, stepCountIs } from 'ai';
import { z } from 'zod';

// The model runs on Modal and scales to zero, so the first message after an
// idle period waits for a cold start (snapshot restore + model wake-up).
export const maxDuration = 300;

// vLLM's OpenAI-compatible server for the open-weight model (chat-llm/serve.py).
// CHAT_LLM_BASE_URL is the deployed Modal URL ending in /v1. The endpoint
// requires a Modal proxy auth token, sent as the Modal-Key / Modal-Secret headers.
const MODAL_AUTH_HEADERS = {
  'Modal-Key': process.env.CHAT_LLM_MODAL_KEY ?? '',
  'Modal-Secret': process.env.CHAT_LLM_MODAL_SECRET ?? '',
};

const modalLLM = createOpenAICompatible({
  name: 'modal',
  baseURL: process.env.CHAT_LLM_BASE_URL,
  headers: MODAL_AUTH_HEADERS,
});

const SYSTEM_PROMPT = `
You are Marvin Wu (Wu Hung Mao) — a CS Specialist at University of Toronto Mississauga, graduating December 2026.
You are NOT an AI assistant — you ARE Marvin, having a casual conversation with visitors to your portfolio website.

## Tone
- Casual but knowledgeable
- Keep responses brief (2–4 short paragraphs)
- End most responses with a follow-up question to keep the conversation going
- When asked something completely unrelated to your portfolio (math, cooking, weather, world events, philosophy, etc.), use the getOffTopic tool and reply with just one ultra-short funny line — no explanation. The tool card handles the redirect.
- Match the language of the user

## Background
- Honours BSc Computer Science Specialist with a Mathematics minor, UofT Mississauga, GPA 3.44/4.0 cumulative (3.54 last two years), graduating December 2026 with PEY CO-OP
- Research interest: efficient, interpretable video understanding for medical imaging, with weak supervision and selective attention
- Current (Fall 2026):
  - Research: Early Gastric Neoplasia Grading from Endoscopy Videos (Oct 2026–present), University of Toronto, supervisor Dr. Ameya Pore — with clinicians, building an annotated upper-GI endoscopy video dataset (real-time hospital-server demo planned). Integrated MedSAM2 into CVAT as a GPU-backed interactor and mask tracker; enabled AMD ROCm GPU acceleration for CVAT's containerized models and ported MedSAM2/TransT to GPU; built a multi-user tracking service (job queue, live progress, cancellation, CVAT auth) that processes a 3,790-frame video in ~70 s
  - Research: Diffusion models (Sep 2026–present), University of Toronto, supervisor Prof. Samir Hamdi — research plan and scope still being defined
  - Teaching Assistant for CSC105 (AI Literacy), supervisor Prof. Josh Jung — office hours/Piazza for ~215 students, teach 4-hour weekly labs for a 40-student section
  - CSC490 capstone: Multi-Camera Fall Detection (4-person team) — several phone cameras stream to one room and vote on each alarm; deployed on Hugging Face Spaces with push alerts. Fine-tuned VideoMAE on EDF, MCFD, UP-Fall, int8-quantized for CPU: caught 93% of falls a median 1.2 s after onset; two-camera agreement cut false alarms 24 → 2 while catching 94% of falls. Fine-tuned Gemma 3 12B with LoRA (H100s on the Fir cluster) as an explaining verifier: 93.8–96.7% balanced accuracy, removed 5 of 7 false alarms, kept 132 of 136 falls. Demo: https://youtu.be/GD_KssecW6U · Code: https://github.com/soloadio/CSC490-Fall-Detection-Capstone-Project
  - CSC413 project: Weakly Supervised Polyp Detection in Colonoscopy Video — model lead on a 3-person team proposing a video transformer that keeps only the top-k most lesion-like patches per frame before temporal attention, trained from video-level labels (~1,000× less attention cost in theory); NeurIPS-format proposal, a testbed for the gastric research
  - Courses in progress: CSC413 (Neural Networks & Deep Learning), CSC478 (Robotic Perception), CSC490 (Capstone)
- 16-month PEY CO-OP at Ericsson (May 2024 – Sep 2025) as Physical Layer (vDU) Developer Intern, Ottawa
  - Gphy2.0: worked with senior engineers to move LDPC encoding from an Intel accelerator to NVIDIA GPUs with CUDA; profiled with Nsight Systems & Compute across A10, A100, and Grace Hopper servers
  - Gemini: 5G digital twin — Vue3 constellation diagram UI, multi-threaded CUDA unit tests under ~35 μs constraint, SQLite + LTTng channel quality pipeline
- AI-Video Detection Ensemble (Apr 2026): senior developer on a 5-person team (recruited three teammates); multimodal ensemble of EfficientNet-B1, MesoNet, XceptionNet, AASIST; ~70% accuracy on unseen FakeAVCeleb vs 90%+ in-domain, gap attributed to domain shift. Deployed by Marvin with a Vercel frontend (built by teammate Yousef) and Modal serverless GPU backend: https://deepfake-detector-gamma.vercel.app/
- Publications:
  - Peer-reviewed: Khan, L., Wu, H.-M., Lin, W., Bi, F., Abdelhadi, Y., Jung, J. "Ensemble Deep Learning Approaches for AI-Altered Video Detection." Journal of Computing, Data, and Exploration (CODEX), vol. 2, no. 1, 2026. arXiv:2607.06872
  - Course research manuscript (CSC398, Apr 2026): "Can Agents Debug Better Than a Software Engineer?" — evaluating LLM coding agents on the JetBrains LCA benchmark
- Sycophancy Bias Replication (TAISI Intensive, Jul 2026): benchmarked Claude, GPT, and Gemini via OpenRouter on "suggested-answer" bias using MMLU, TruthfulQA, LogiQA, HellaSwag questions
- Contributed to JetBrains LCA benchmark — extended bug localization to automated fix application, presented to 40+ attendees
- Contact: hongmao.wu@mail.utoronto.ca | linkedin.com/in/hungmao-wu | github.com/wuhungmao

## Skills
- Languages: Python, C, C++, CUDA, Java, SQL, JavaScript, HTML/CSS, PHP
- Frameworks: React, Node.js, jQuery, Django, Express.js, Vue3
- ML & CV: PyTorch, OpenCV, NumPy, Pandas, Hugging Face, Albumentations
- Tools: Linux, Git, Docker, AWS, Jenkins, Jira, Bazel, Gerrit, Modal, Vercel
- Databases: PostgreSQL, SQLite
- HPC/Profiling: CUDA, OpenMP, MPI, Nsight Systems, LTTng, Valgrind

## Tool Usage Guidelines
- Use AT MOST ONE tool per response
- The tool renders a visual card/component on the page — don't repeat its contents in your text
- Just introduce it briefly and let the component speak for itself
- Use getProjects when asked about projects, what you've built, your work, coding
- Use getSkills when asked about skills, tech stack, languages, tools, what you know
- Use getExperience when asked about work experience, internship, Ericsson, job, co-op, TA, teaching, research position
- Use getPublications when asked about research, papers, publications
- Use getTimeline when asked about your journey, education, history, timeline, background
- Use getContact when asked how to contact you, your email, LinkedIn, GitHub, hire you
- Use getHobbies when asked about hobbies, interests, what you do for fun, outside of work
- Use getRecentActivity when asked what you've been up to lately, recent coding, current projects, what you're working on right now
- Use getOffTopic when asked anything unrelated to you — math, trivia, news, cooking, philosophy, relationships, etc.
`;

const tools = {
  getProjects: tool({
    description: "Show a list of Marvin's projects",
    parameters: z.object({}),
    execute: async () => "Rendering projects list.",
  }),
  getSkills: tool({
    description: "Show Marvin's technical skills and tech stack",
    parameters: z.object({}),
    execute: async () => "Rendering skills.",
  }),
  getExperience: tool({
    description: "Show Marvin's work and research experience (research, TA, Ericsson)",
    parameters: z.object({}),
    execute: async () => "Rendering work experience.",
  }),
  getPublications: tool({
    description: "Show Marvin's research publications",
    parameters: z.object({}),
    execute: async () => "Rendering publications.",
  }),
  getTimeline: tool({
    description: "Show Marvin's education and career timeline",
    parameters: z.object({}),
    execute: async () => "Rendering timeline.",
  }),
  getContact: tool({
    description: "Show Marvin's contact information",
    parameters: z.object({}),
    execute: async () => "Rendering contact info.",
  }),
  getHobbies: tool({
    description: "Show Marvin's hobbies and interests",
    parameters: z.object({}),
    execute: async () => "Rendering hobbies.",
  }),
  getOffTopic: tool({
    description: "Shown when the user asks something completely unrelated to Marvin's portfolio",
    parameters: z.object({}),
    execute: async () => "Rendering off-topic response.",
  }),
  getRecentActivity: tool({
    description: "Fetch Marvin's recent GitHub activity to show what he's been coding lately",
    parameters: z.object({}),
    execute: async () => {
      try {
        const res = await fetch('https://api.github.com/users/wuhungmao/events?per_page=20', {
          headers: { 'User-Agent': 'marvin-portfolio-bot' },
        });
        const events = await res.json();
        if (!Array.isArray(events)) return JSON.stringify([]);
        const parsed = events
          .filter(e => ['PushEvent', 'CreateEvent', 'PullRequestEvent', 'ForkEvent'].includes(e.type))
          .slice(0, 8)
          .map(e => ({
            type: e.type,
            repo: e.repo.name.replace('wuhungmao/', ''),
            date: e.created_at,
            message:
              e.type === 'PushEvent'
                ? (e.payload.commits?.[0]?.message?.split('\n')[0] ?? 'pushed changes')
                : e.type === 'CreateEvent'
                ? `created ${e.payload.ref_type}${e.payload.ref ? ': ' + e.payload.ref : ''}`
                : e.type === 'PullRequestEvent'
                ? `${e.payload.action} PR: ${e.payload.pull_request?.title ?? ''}`
                : e.type === 'ForkEvent'
                ? `forked ${e.payload.forkee?.full_name ?? ''}`
                : e.type,
          }));
        return JSON.stringify(parsed);
      } catch {
        return JSON.stringify([]);
      }
    },
  }),
};

// The GitHub Pages copy of the site calls this endpoint cross-origin.
const CORS_HEADERS = {
  'Access-Control-Allow-Origin': 'https://wuhungmao.github.io',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

export function OPTIONS() {
  return new Response(null, { status: 204, headers: CORS_HEADERS });
}

// Warm-up: the chat page calls this when it opens, so the model starts booting
// while the visitor reads and types. vLLM's /health answers once it is ready.
export async function GET() {
  const healthURL = (process.env.CHAT_LLM_BASE_URL ?? '').replace(/\/v1\/?$/, '') + '/health';
  try {
    const res = await fetch(healthURL, {
      headers: MODAL_AUTH_HEADERS,
      signal: AbortSignal.timeout(150_000),
    });
    return new Response(null, { status: res.ok ? 204 : 503, headers: CORS_HEADERS });
  } catch (err) {
    console.error('Chat warm-up error:', err);
    return new Response(null, { status: 503, headers: CORS_HEADERS });
  }
}

export async function POST(request) {
  try {
    const { messages } = await request.json();

    const modelMessages = await convertToModelMessages(messages);
    const result = streamText({
      model: modalLLM.chatModel('marvin-chat'),
      system: SYSTEM_PROMPT,
      messages: modelMessages,
      tools,
      stopWhen: stepCountIs(2),
    });

    return result.toUIMessageStreamResponse({
      headers: CORS_HEADERS,
      // Errors during streaming (e.g. a rejected API key) otherwise reach the
      // client only as a generic message, so log them for the Vercel logs.
      onError: (err) => {
        console.error('Chat stream error:', err);
        return 'Something went wrong while generating a response.';
      },
    });
  } catch (err) {
    console.error('Chat error:', err);
    return new Response(err?.message || 'Internal Server Error', { status: 500, headers: CORS_HEADERS });
  }
}

import { anthropic } from '@ai-sdk/anthropic';
import { streamText, tool, convertToModelMessages } from 'ai';
import { z } from 'zod';

export const maxDuration = 30;

const SYSTEM_PROMPT = `
You are Marvin Wu (Wu Hung Mao) — a CS Specialist at University of Toronto Mississauga, graduating August 2026.
You are NOT an AI assistant — you ARE Marvin, having a casual conversation with visitors to your portfolio website.

## Tone
- Casual but knowledgeable
- Keep responses brief (2–4 short paragraphs)
- End most responses with a follow-up question to keep the conversation going
- When asked something completely unrelated to your portfolio (math, cooking, weather, world events, philosophy, etc.), use the getOffTopic tool and reply with just one ultra-short funny line — no explanation. The tool card handles the redirect.
- Match the language of the user

## Background
- Honours BSc Computer Science Specialist, UofT Mississauga, GPA 3.45/4.0, graduating August 2026 with PEY CO-OP
- 16-month PEY CO-OP at Ericsson (May 2024 – Sep 2025) as Physical Layer (vDU) Developer Intern, Ottawa
  - Gphy2.0: offloaded LDPC encoding from Intel accelerator to Nvidia GPU using CUDA; profiled with Nsight Systems & Compute
  - Gemini: built 5G digital twin — Vue3 constellation diagram UI, multi-threaded CUDA unit tests under ~35 μs constraint, SQLite + LTTng channel quality pipeline
- Led 5-person team building deepfake detection ensemble (EfficientNet-B1, MesoNet, XceptionNet, AASIST) on AIGVDBench & FaceForensics++; co-authored research manuscript
- Contributed to JetBrains LCA benchmark — extended bug localization to automated fix application, co-authored white paper, presented to 40+ attendees
- 2 co-authored research papers (April 2026)
- Contact: hongmao.wu@mail.utoronto.ca | linkedin.com/in/hungmao-wu | github.com/wuhungmao

## Skills
- Languages: Python, C, C++, Java, SQL, JavaScript, HTML/CSS, PHP
- Frameworks: React, Node.js, jQuery, Django, Express.js, Vue3
- Tools: Linux, Git, Docker, AWS, Jenkins, Jira, Bazel, Gerrit
- Databases: PostgreSQL, SQLite
- HPC/Profiling: CUDA, OpenMP, MPI, Nsight Systems, LTTng, Valgrind

## Tool Usage Guidelines
- Use AT MOST ONE tool per response
- The tool renders a visual card/component on the page — don't repeat its contents in your text
- Just introduce it briefly and let the component speak for itself
- Use getProjects when asked about projects, what you've built, your work, coding
- Use getSkills when asked about skills, tech stack, languages, tools, what you know
- Use getExperience when asked about work experience, internship, Ericsson, job, co-op
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
    description: "Show Marvin's work experience at Ericsson",
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

export async function POST(request) {
  try {
    const { messages } = await request.json();

    const modelMessages = await convertToModelMessages(messages);
    const result = streamText({
      model: anthropic('claude-haiku-4-5-20251001'),
      system: SYSTEM_PROMPT,
      messages: modelMessages,
      tools,
      maxSteps: 2,
    });

    return result.toUIMessageStreamResponse();
  } catch (err) {
    console.error('Chat error:', err);
    return new Response(err?.message || 'Internal Server Error', { status: 500 });
  }
}

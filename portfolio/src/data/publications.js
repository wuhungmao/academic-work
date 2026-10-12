const publications = [
  {
    id: 'ai-video-paper',
    title: 'Ensemble Deep Learning Approaches for AI-Altered Video Detection',
    authors: 'Khan, L., Wu, H.-M., Lin, W., Bi, F., Abdelhadi, Y., Jung, J.',
    venue: 'Journal of Computing, Data, and Exploration (CODEX), vol. 2, no. 1, 2026 · Preprint: arXiv:2607.06872',
    date: 'Sep 2026',
    status: 'published',
    description: 'Presents a multimodal deepfake-detection ensemble — AASIST for audio; EfficientNet-B1, XceptionNet, and MesoNet for video frames — trained on AIGVDBench, FaceForensics++, and ASVspoof 2019 and evaluated on FakeAVCeleb, comparing five fusion strategies.',
    pdf: 'https://doi.org/10.33137/codex.v2i1.47792',
    tags: ['Deep Learning', 'Deepfake Detection', 'Ensemble Methods', 'Computer Vision', 'Peer-reviewed'],
  },
  {
    id: 'agents-debug-paper',
    title: 'Can Agents Debug Better Than a Software Engineer?',
    authors: 'Riku, T., Ahmed, K., Wu, H.-M., Usman, R.',
    venue: 'Course Research Manuscript (CSC398), University of Toronto Mississauga',
    date: 'Apr 2026',
    status: 'unpublished',
    description: 'Evaluates the bug-localization and automated fix capabilities of LLM-based coding agents (Claude Code, Codex) against the JetBrains LCA benchmark. Extends the benchmark from file-level bug localization to automated patch generation, and introduces expanded evaluation metrics including token usage, iteration count, and fix success rate.',
    pdf: 'https://drive.google.com/file/d/1YPqEK_rhctJtD1Wj-wzSBSfd9r5xmhBI/view?usp=sharing',
    tags: ['LLM Agents', 'Software Engineering', 'Benchmarking', 'Claude Code', 'Codex'],
  },
]

export default publications

const projects = [
  {
    id: 'gpu-image',
    name: 'Image Processing with GPU',
    course: 'CSC367',
    description: 'Implemented image processing algorithms using CUDA, achieving significant speedups over CPU implementations through parallel execution on GPU cores.',
    tags: ['CUDA', 'C', 'Parallel Computing', 'GPU'],
    pdf: '/academic-work/CSC367/Image%20processing%20with%20GPU/report.pdf',
  },
  {
    id: 'db-join',
    name: 'Database Join with OpenMP & MPI',
    course: 'CSC367',
    description: 'Parallelized relational database join operations using OpenMP for shared-memory and MPI for distributed-memory parallelism, with performance benchmarking.',
    tags: ['OpenMP', 'MPI', 'C', 'Parallel Computing'],
    pdf: '/academic-work/CSC367/Database%20join%20with%20openMP%20and%20MPI/report.pdf',
  },
  {
    id: 'simon-game',
    name: 'Simon Game',
    course: 'CSC258',
    description: 'Built a Simon memory game in MIPS assembly language running on a MARS simulator, implementing game logic, display output, and user input at the hardware level.',
    tags: ['MIPS Assembly', 'Computer Organization', 'Hardware'],
    pdf: '/academic-work/CSC258/Simon%20game%20guide.pdf',
  },
  {
    id: 'boggle',
    name: 'Boggle Game',
    course: 'CSC207',
    description: 'Developed a full Boggle word game in Java using object-oriented design patterns including MVC architecture, following SOLID principles throughout.',
    tags: ['Java', 'OOP', 'MVC', 'Design Patterns'],
    pdf: '/academic-work/CSC207/project_phase1_template.pdf',
  },
]

export default projects

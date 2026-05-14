import useInView from '../hooks/useInView'

export default function FadeIn({ children, delay = 0, style = {}, className = '' }) {
  const [ref, inView] = useInView()
  return (
    <div
      ref={ref}
      className={`fade-in-up ${inView ? 'visible' : ''} ${className}`}
      style={{ transitionDelay: `${delay}ms`, ...style }}
    >
      {children}
    </div>
  )
}

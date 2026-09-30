const base = {
  width: 24,
  height: 24,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.6,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  'aria-hidden': true,
}

const make = (children) =>
  function Icon(props) {
    return <svg {...base} {...props}>{children}</svg>
  }

export const Gateway = make(<><path d="M4 12h10" /><path d="M10 7l5 5-5 5" /><path d="M19 4v16" /></>)
export const Shield = make(<><path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z" /><path d="M9 12l2 2 4-4" /></>)
export const Stack = make(<><rect x="4" y="4" width="16" height="4" rx="1" /><rect x="4" y="10" width="16" height="4" rx="1" /><rect x="4" y="16" width="10" height="4" rx="1" /></>)
export const Gauge = make(<><path d="M4 16a8 8 0 1116 0" /><path d="M12 16l4-5" /><circle cx="12" cy="16" r="1" /></>)
export const Plug = make(<><path d="M9 3v5M15 3v5" /><path d="M6 8h12v3a6 6 0 01-12 0z" /><path d="M12 17v4" /></>)
export const Chip = make(<><rect x="6" y="6" width="12" height="12" rx="2" /><rect x="9.5" y="9.5" width="5" height="5" /><path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3" /></>)
export const Play = make(<path d="M8 5l11 7-11 7z" fill="currentColor" />)
export const Pause = make(<><path d="M8 5v14M16 5v14" strokeWidth="3" /></>)
export const Replay = make(<><path d="M4 12a8 8 0 108-8" /><path d="M4 4v5h5" /></>)
export const Arrow = make(<><path d="M5 12h14" /><path d="M13 6l6 6-6 6" /></>)
export const Github = make(<path d="M9 19c-4 1.2-4-2-6-2.5M15 21v-3.2a2.8 2.8 0 00-.8-2.2c2.7-.3 5.5-1.3 5.5-6a4.6 4.6 0 00-1.3-3.2 4.3 4.3 0 00-.1-3.2s-1-.3-3.4 1.3a11.7 11.7 0 00-6 0C6.5 2.9 5.5 3.2 5.5 3.2a4.3 4.3 0 00-.1 3.2A4.6 4.6 0 004 9.6c0 4.6 2.8 5.7 5.5 6A2.8 2.8 0 008.700 17.800V21" />)
export const Linkedin = make(<><rect x="3" y="3" width="18" height="18" rx="3" /><path d="M8 11v5M8 8v.01M12 16v-5M12 13a2.5 2.5 0 015 0v3" /></>)

export const STAGE_ICONS = { gateway: Gateway, admission: Shield, queue: Stack, controller: Gauge, backend: Plug, gpu: Chip }

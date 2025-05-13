
import type { SVGProps } from 'react';

export function MailThreaderLogo(props: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor" // Default stroke for paths/lines unless overridden
      strokeWidth="2"   // Default stroke-width
      strokeLinecap="round"
      strokeLinejoin="round"
      xmlns="http://www.w3.org/2000/svg"
      {...props} // Spread props to allow className, etc.
    >
      {/* Envelope shape */}
      <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
      <polyline points="22,6 12,13 2,6" />

      {/* Thread: three circles connected by a line, representing sequence/connection */}
      {/* Placed on the main body of the envelope */}
      <line x1="7" y1="15" x2="17" y2="15" strokeWidth="2" /> {/* Connecting line */}
      {/* Circles on the thread. fill="currentColor" makes them solid. stroke="none" ensures no extra border on them. */}
      <circle cx="7" cy="15" r="1.5" fill="currentColor" stroke="none" />
      <circle cx="12" cy="15" r="1.5" fill="currentColor" stroke="none" />
      <circle cx="17" cy="15" r="1.5" fill="currentColor" stroke="none" />
    </svg>
  );
}

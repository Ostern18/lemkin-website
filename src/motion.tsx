// motion.tsx
import React from "react";
import { motion, AnimatePresence, useIsPresent } from "framer-motion";

/** ----------------------------------------------------------------
 *  Timing & Easing (enterprise-friendly defaults)
 *  ---------------------------------------------------------------- */
export const timings = {
  fast: 0.15,
  base: 0.25,
  slow: 0.40,
} as const;

export const ease = {
  out: [0.16, 1, 0.3, 1],      // easeOutExpo-ish, crisp but calm
  inOut: [0.12, 0, 0.39, 0],   // easeInOut for modal/panels
} as const;

export const springs = {
  press: { type: "spring" as const, stiffness: 420, damping: 24, mass: 0.9 },
  hover: { type: "spring" as const, stiffness: 260, damping: 22 },
  layout: { type: "spring" as const, stiffness: 240, damping: 26 },
};

/** ----------------------------------------------------------------
 *  Reusable Variants
 *  ---------------------------------------------------------------- */
// Page/section fade-up on first paint
export const fadeUp = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0, transition: { duration: timings.slow, ease: ease.out } },
  exit: { opacity: 0, y: 16, transition: { duration: timings.base } },
};

// Simple fade
export const fade = {
  initial: { opacity: 0 },
  animate: { opacity: 1, transition: { duration: timings.base, ease: ease.out } },
  exit: { opacity: 0, transition: { duration: timings.fast } },
};

// Elevation card hover (no tilt to keep it serious)
export const elevate = {
  rest:   { y: 0, scale: 1, boxShadow: "0 1px 3px rgba(0,0,0,0.1)", transition: springs.layout },
  hover:  { y: -4, scale: 1.01, boxShadow: "0 10px 25px rgba(0,0,0,0.15)", transition: springs.hover },
  tap:    { scale: 0.99, transition: springs.press },
};

// Modal enter/exit (slide-up + fade)
export const modal = {
  backdrop: {
    initial: { opacity: 0 },
    animate: { opacity: 1, transition: { duration: timings.base } },
    exit:    { opacity: 0, transition: { duration: timings.fast } },
  },
  panel: {
    initial: { opacity: 0, y: 24, scale: 0.98 },
    animate: { opacity: 1, y: 0, scale: 1, transition: { duration: timings.base, ease: ease.out } },
    exit:    { opacity: 0, y: 24, scale: 0.98, transition: { duration: timings.base, ease: ease.inOut } },
  },
};

// Staggered list/container
export const stagger = (delay = 0, step = 0.06) => ({
  hidden: { opacity: 0, y: 8 },
  show: {
    opacity: 1, y: 0,
    transition: { delay, staggerChildren: step, ease: ease.out, duration: timings.base }
  },
});

// Child item for stagger
export const item = {
  hidden: { opacity: 0, y: 8 },
  show: { opacity: 1, y: 0, transition: { ease: ease.out, duration: timings.base } },
};

/** ----------------------------------------------------------------
 *  Primitives & Wrappers
 *  ---------------------------------------------------------------- */

// Pressable button wrapper (adds tactile press)
export const Pressable: React.FC<React.ComponentProps<typeof motion.button>> = ({ children, ...props }) => (
  <motion.button whileTap={{ scale: 0.96 }} transition={springs.press} {...props}>
    {children}
  </motion.button>
);

// Elevating card wrapper
export const MotionCard: React.FC<React.ComponentProps<typeof motion.div>> = ({ children, ...props }) => (
  <motion.div
    initial="rest"
    whileHover="hover"
    whileTap="tap"
    variants={elevate}
    {...props}
  >
    {children}
  </motion.div>
);

// Modal shell (backdrop + panel)
export const ModalShell: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  labelledBy?: string;
}> = ({ isOpen, onClose, children, labelledBy = "modal-title" }) => (
  <AnimatePresence>
    {isOpen && (
      <motion.div
        key="backdrop"
        className="fixed inset-0 z-50 bg-black/40"
        initial="initial"
        animate="animate"
        exit="exit"
        variants={modal.backdrop}
        onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
        aria-hidden
      >
        <motion.div
          key="panel"
          role="dialog"
          aria-modal="true"
          aria-labelledby={labelledBy}
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          variants={modal.panel}
        >
          <div className="bg-[var(--surface-2)] border border-[var(--bd)] rounded-2xl shadow-[var(--shadow-lg)] max-w-4xl w-full max-h-[80vh] overflow-auto">
            {children}
          </div>
        </motion.div>
      </motion.div>
    )}
  </AnimatePresence>
);

// Skeleton (shimmer) block
export const Skeleton: React.FC<{ className?: string }> = ({ className = "" }) => (
  <div className={`relative overflow-hidden bg-[var(--surface-1)] ${className}`}>
    <div className="absolute inset-0 -translate-x-full animate-shimmer
                    bg-gradient-to-r from-transparent via-white/40 dark:via-white/10 to-transparent" />
  </div>
);

// Page transition bar (top progress) — controlled by parent
export const RouteProgressBar: React.FC<{ progress: number }> = ({ progress }) => {
  const isPresent = useIsPresent();
  return (
    <motion.div
      aria-hidden
      className="fixed top-0 left-0 h-[2px] z-[60] bg-[var(--accent)]"
      style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
      initial={{ width: 0 }}
      animate={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
      exit={{ width: 0, opacity: 0, transition: { duration: timings.fast } }}
      transition={{ duration: timings.fast, ease: ease.out }}
    >
      {/* tiny tail blur for speed perception */}
      <motion.div
        className="h-full w-6 bg-[var(--accent)]/60 blur-[2px]"
        initial={{ opacity: 0 }}
        animate={{ opacity: isPresent ? 1 : 0 }}
        transition={{ duration: timings.fast }}
      />
    </motion.div>
  );
};

/** ----------------------------------------------------------------
 *  Convenience Components for Common Patterns
 *  ---------------------------------------------------------------- */

// SectionFadeUp wraps any section with first-paint fade-up
export const SectionFadeUp: React.FC<{
  children: React.ReactNode;
  className?: string;
  as?: keyof JSX.IntrinsicElements
}> = ({
  children, className = "", as: Tag = "section",
}) => (
  <motion.section variants={fadeUp} initial="initial" animate="animate" exit="exit" className={className}>
    {children}
  </motion.section>
);

// StaggeredList wraps a container and applies stagger to its children
export const StaggeredList: React.FC<{
  children: React.ReactNode;
  className?: string
}> = ({ children, className = "" }) => (
  <motion.div variants={stagger()} initial="hidden" animate="show" className={className}>
    {React.Children.map(children, (child) => (
      <motion.div variants={item}>{child}</motion.div>
    ))}
  </motion.div>
);
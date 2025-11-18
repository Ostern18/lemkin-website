import * as React from "react"
import { motion } from "framer-motion"
import { cn } from "../../lib/utils"
import { LucideIcon } from "lucide-react"

interface MenuItem {
  icon: LucideIcon | React.FC
  label: string
  href: string
  gradient: string
  iconColor: string
}

interface MenuBarProps {
  items: MenuItem[]
  activeItem?: string
  onItemClick?: (label: string) => void
  isDarkTheme?: boolean
  className?: string
}

export const MenuBar = React.forwardRef<HTMLDivElement, MenuBarProps>(
  ({ className, items, activeItem, onItemClick, isDarkTheme = false, ...props }, ref) => {
    return (
      <nav
        ref={ref}
        className={cn(
          "p-2 rounded-2xl bg-gradient-to-b from-[var(--color-bg-primary)]/80 to-[var(--color-bg-primary)]/40 backdrop-blur-lg border border-[var(--color-border-primary)]/40 shadow-lg relative overflow-hidden",
          className,
        )}
        {...props}
      >
        <motion.div
          className={`absolute -inset-2 rounded-3xl z-0 pointer-events-none transition-opacity duration-500 ${
            isDarkTheme
              ? "bg-gradient-radial from-blue-400/20 via-purple-400/15 to-transparent"
              : "bg-gradient-radial from-blue-400/10 via-purple-400/8 to-transparent"
          }`}
          initial={{ opacity: 0 }}
          whileHover={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        />
        <ul className="flex items-center gap-2 relative z-10">
          {items.map((item) => {
            const Icon = item.icon
            const isActive = item.label === activeItem

            return (
              <motion.li key={item.label} className="relative">
                <button
                  onClick={() => onItemClick?.(item.label)}
                  className="block w-full"
                >
                  <motion.div
                    className="block rounded-xl overflow-hidden group relative perspective-600"
                    whileHover={{ scale: 1.02 }}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                  >
                    {/* Glow background */}
                    <motion.div
                      className="absolute inset-0 z-0 pointer-events-none rounded-xl"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: isActive ? 1 : 0 }}
                      whileHover={{ opacity: 1 }}
                      transition={{ duration: 0.3 }}
                      style={{
                        background: item.gradient,
                        filter: "blur(8px)",
                      }}
                    />
                    
                    {/* Main content */}
                    <motion.div
                      className={cn(
                        "flex items-center gap-2 px-4 py-2 relative z-10 bg-transparent transition-colors rounded-xl",
                        isActive
                          ? "text-[var(--color-text-primary)]"
                          : "text-[var(--color-text-secondary)] group-hover:text-[var(--color-text-primary)]",
                      )}
                      transition={{ type: "spring", stiffness: 200, damping: 20 }}
                    >
                      <span
                        className={cn(
                          "transition-colors duration-300",
                          isActive ? item.iconColor : "text-[var(--color-text-primary)]",
                        )}
                        style={{
                          color: isActive ? undefined : 'var(--color-text-primary)',
                          ...(isActive && { color: item.iconColor.replace('text-', '') })
                        }}
                      >
                        <Icon className="h-5 w-5" />
                      </span>
                      <span className="font-medium">{item.label}</span>
                    </motion.div>
                  </motion.div>
                </button>
              </motion.li>
            )
          })}
        </ul>
      </nav>
    )
  },
)

MenuBar.displayName = "MenuBar"
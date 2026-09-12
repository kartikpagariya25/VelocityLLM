import { motion } from 'framer-motion';

const sectionVariants = {
  hidden: { opacity: 0, y: 60 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.7,
      ease: [0.16, 1, 0.3, 1],
      staggerChildren: 0.12,
    },
  },
};

const childVariants = {
  hidden: { opacity: 0, y: 40 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] },
  },
};

export default function SectionWrapper({ children, className = '', id = '' }) {
  return (
    <motion.section
      id={id}
      className={`section ${className}`}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.15 }}
      variants={sectionVariants}
    >
      {children}
    </motion.section>
  );
}

export { sectionVariants, childVariants };
export const MotionDiv = ({ children, className = '', delay = 0, ...props }) => (
  <motion.div
    className={className}
    variants={{
      hidden: { opacity: 0, y: 40 },
      visible: {
        opacity: 1,
        y: 0,
        transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1], delay },
      },
    }}
    {...props}
  >
    {children}
  </motion.div>
);

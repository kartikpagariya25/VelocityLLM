import { motion } from 'framer-motion';
import './ModelStrip.css';

const models = [
  'TinyLlama-1.1B',
  'Llama-3.2-1B-Instruct',
  'StableLM-2-1.6B',
  'Qwen2.5-1.5B-Instruct',
];

export default function ModelStrip() {
  return (
    <div className="model-strip">
      <div className="model-strip__container">
        <div className="model-strip__label">
          Model Agnostic
          <span className="model-strip__label-arrow">→</span>
        </div>
        <div className="model-strip__track-wrapper">
          <motion.div
            className="model-strip__track"
            animate={{
              x: ['0%', '-50%'],
            }}
            transition={{
              repeat: Infinity,
              ease: 'linear',
              duration: 20,
            }}
          >
            {/* Double the array for seamless infinite scrolling */}
            {[...models, ...models].map((model, i) => (
              <div key={i} className="model-strip__item">
                {model}
              </div>
            ))}
          </motion.div>
        </div>
      </div>
    </div>
  );
}

// Minimal AI Gateway check: generate text through the gateway.
//   node --env-file=../.env.local --experimental-strip-types index.ts
import { generateText } from 'ai';

const { text } = await generateText({
  model: 'openai/gpt-5.5',
  prompt: 'Invent a new holiday and describe its traditions in one short paragraph.',
});

console.log(text);

// Sidecar that exposes Jev (via Vercel AI Gateway) over local HTTP, so the Python
// evaluation harness can call it with the same question specs it uses everywhere else.
//
//   node --env-file=../.env.local --experimental-strip-types jev-server.ts
//   POST /evaluate {"state": "...", "questions": [{key,type,question,options|legend}]}
//
// Question specs are exactly `system_one.types.to_spec()` output.
import { createServer } from 'node:http';
import { experimental_evaluate as evaluate } from 'ai';

const MODEL = process.env.JEV_MODEL ?? 'typesafe-ai/jev';
const PORT = Number(process.env.JEV_PORT ?? 8099);

type Spec = {
  key: string;
  type: 'bool' | 'choice' | 'score';
  question: string;
  options?: string[];
  legend?: string[];
};

function toQuestions(specs: Spec[]) {
  const questions: Record<string, unknown> = {};
  for (const spec of specs) {
    if (spec.type === 'bool') {
      questions[spec.key] = { type: 'boolean', instructions: spec.question };
    } else if (spec.type === 'choice') {
      const criteria: Record<string, null> = {};
      for (const option of spec.options ?? []) criteria[option] = null;
      questions[spec.key] = { type: 'choice', instructions: spec.question, criteria };
    } else {
      questions[spec.key] = {
        type: 'score',
        instructions: spec.question,
        criteria: (spec.legend ?? []).map((level) => level ?? null),
      };
    }
  }
  return questions;
}

// Normalize Jev's answers into the same shape system_one's own types produce.
function toAnswers(specs: Spec[], answers: Record<string, any>) {
  const out: Record<string, unknown> = {};
  for (const spec of specs) {
    const a = answers[spec.key];
    if (!a) continue;
    if (a.type === 'boolean') {
      out[spec.key] = { type: 'bool', p_true: a.probability, value: a.probability >= 0.5 };
    } else if (a.type === 'choice') {
      const probs = a.probabilities ?? {};
      out[spec.key] = {
        type: 'choice',
        choice: a.choice,
        confidence: probs[a.choice] ?? 1,
        probabilities: Object.fromEntries((spec.options ?? []).map((o) => [o, probs[o] ?? 0])),
      };
    } else {
      const probs = a.probabilities ?? {};
      const levels = (spec.legend ?? []).map((_, i) => probs[String(i)] ?? 0);
      const level = levels.indexOf(Math.max(...levels));
      out[spec.key] = {
        type: 'score',
        score: a.score,
        level: level < 0 ? Math.round(a.score) : level,
        confidence: levels.length ? Math.max(...levels) : 1,
        legend: Object.fromEntries((spec.legend ?? []).map((d, i) => [i, d])),
      };
    }
  }
  return out;
}

const server = createServer(async (req, res) => {
  if (req.method === 'GET' && req.url === '/health') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ ok: true, model: MODEL }));
    return;
  }
  if (req.method !== 'POST' || req.url !== '/evaluate') {
    res.writeHead(404).end();
    return;
  }
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  try {
    const body = JSON.parse(Buffer.concat(chunks).toString());
    const specs: Spec[] = body.questions;
    const started = performance.now();
    const result = await evaluate({
      model: MODEL,
      state: body.state,
      questions: toQuestions(specs) as any,
    });
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(
      JSON.stringify({
        answers: toAnswers(specs, result.answers as Record<string, any>),
        latency_ms: performance.now() - started,
        usage: result.usage,
        model: result.response?.modelId ?? MODEL,
      }),
    );
  } catch (error: any) {
    res.writeHead(500, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: String(error?.message ?? error), name: error?.name }));
  }
});

server.listen(PORT, () => console.log(`jev sidecar on http://localhost:${PORT} using ${MODEL}`));

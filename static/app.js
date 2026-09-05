const source = document.querySelector('#source');
const analyzeButton = document.querySelector('#analyze');
const status = document.querySelector('#status');
const results = document.querySelector('#results');
const chatForm = document.querySelector('#chat-form');
const messages = document.querySelector('#messages');
let jobId;

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle('error', isError);
}

function text(id, value) { document.querySelector(`#${id}`).textContent = value || 'No information found.'; }

function showResult(result) {
  text('title', result.title);
  text('summary', result.summary);
  text('actions', result.actions);
  text('decisions', result.decisions);
  text('questions', result.questions);
  results.classList.remove('hidden');
}

async function poll() {
  const response = await fetch(`/api/jobs/${jobId}`);
  const job = await response.json();
  if (job.status === 'processing') {
    setStatus(job.message);
    window.setTimeout(poll, 2000);
  } else if (job.status === 'complete') {
    setStatus(job.message);
    showResult(job.result);
    analyzeButton.disabled = false;
  } else {
    setStatus(job.error || 'Processing failed.', true);
    analyzeButton.disabled = false;
  }
}

analyzeButton.addEventListener('click', async () => {
  if (!source.value.trim()) return setStatus('Enter a YouTube URL or local file path.', true);
  analyzeButton.disabled = true;
  results.classList.add('hidden');
  setStatus('Starting analysis…');
  const response = await fetch('/api/jobs', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({source: source.value.trim()}),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.detail || 'Could not start analysis.', true);
    analyzeButton.disabled = false;
    return;
  }
  jobId = data.id;
  poll();
});

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const input = document.querySelector('#question');
  const question = input.value.trim();
  if (!question || !jobId) return;
  messages.insertAdjacentHTML('beforeend', `<p class="question">${escapeHtml(question)}</p>`);
  input.value = '';
  const response = await fetch(`/api/jobs/${jobId}/chat`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question}),
  });
  const data = await response.json();
  messages.insertAdjacentHTML('beforeend', `<p class="answer">${escapeHtml(data.answer || data.detail || 'No answer available.')}</p>`);
});

function escapeHtml(value) {
  const element = document.createElement('div'); element.textContent = value; return element.innerHTML;
}

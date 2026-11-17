(() => {
  const status = document.getElementById('train-status');
  if (status) {
    const poll = async () => {
      try {
        const response = await fetch(status.dataset.url, {headers: {'Accept': 'application/json'}});
        if (!response.ok) return;
        const payload = await response.json();
        status.innerHTML = payload.html;
      } catch (_) {
        status.textContent = 'Status temporarily unavailable';
      }
    };
    poll();
    window.setInterval(poll, 3000);
  }

  if (document.querySelector('[data-eval-running="true"]')) {
    window.setTimeout(() => window.location.reload(), 3000);
  }

  document.querySelectorAll('form').forEach((form) => {
    form.addEventListener('submit', () => {
      const button = form.querySelector('button[type="submit"]');
      if (!button || form.dataset.submitting) return;
      form.dataset.submitting = 'true';
      button.setAttribute('aria-busy', 'true');
      button.classList.add('is-working');
    });
  });
})();

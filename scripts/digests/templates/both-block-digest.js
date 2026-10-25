// --- Interactive Table Filtering ---
document.querySelectorAll('.tbl-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const filter = btn.getAttribute('data-tbl-filter');
    btn.parentElement.querySelectorAll('.tbl-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const table = document.querySelector('.master-table');
    const rows = table.querySelectorAll('tbody tr');

    rows.forEach(tr => {
      if (filter === 'all') {
        tr.style.display = '';
      } else {
        if (tr.classList.contains(`group-${filter}`)) {
          tr.style.display = '';
        } else {
          tr.style.display = 'none';
        }
      }
    });
  });
});

// --- Interactive Chart Filtering & Soloing ---
document.querySelectorAll('.chart-wrapper').forEach(wrapper => {
  const chartId = wrapper.getAttribute('data-chart');
  const svg = wrapper.querySelector('svg');
  const tooltip = wrapper.querySelector('.chart-tooltip');
  const allSeries = svg.querySelectorAll('.chart-series');
  const allBadges = wrapper.querySelectorAll('.legend-badge');
  let soloSeriesId = null;

  function highlightSeries(seriesId) {
    if (soloSeriesId && soloSeriesId !== seriesId) return;

    allSeries.forEach(s => {
      if (s.getAttribute('data-series-id') === seriesId) {
        s.classList.remove('dimmed');
        s.classList.add('highlighted');
        // bring to front
        svg.appendChild(s);
      } else {
        s.classList.remove('highlighted');
        s.classList.add('dimmed');
      }
    });

    allBadges.forEach(b => {
      if (b.getAttribute('data-series-id') === seriesId) {
        b.classList.remove('dimmed');
        b.classList.add('active');
      } else {
        b.classList.remove('active');
        b.classList.add('dimmed');
      }
    });
  }

  function resetHighlight() {
    if (soloSeriesId) {
      highlightSeries(soloSeriesId);
      return;
    }
    allSeries.forEach(s => {
      s.classList.remove('dimmed');
      s.classList.remove('highlighted');
    });
    allBadges.forEach(b => {
      b.classList.remove('dimmed');
      b.classList.remove('active');
    });
    if (tooltip) tooltip.style.display = 'none';
  }

  // Hover over series group in SVG
  allSeries.forEach(s => {
    s.addEventListener('mouseenter', () => {
      highlightSeries(s.getAttribute('data-series-id'));
    });
    s.addEventListener('mouseleave', () => {
      resetHighlight();
    });
  });

  // Hover over data point in SVG
  svg.querySelectorAll('.series-point').forEach(pt => {
    pt.addEventListener('mouseenter', (e) => {
      const sid = pt.parentElement.getAttribute('data-series-id');
      highlightSeries(sid);

      const epoch = pt.getAttribute('data-epoch');
      const val = pt.getAttribute('data-val');
      const seriesName = pt.getAttribute('data-series');
      const group = pt.getAttribute('data-group');

      tooltip.innerHTML = `
        <div style="font-weight:700;font-size:12.5px;margin-bottom:2px;">${seriesName}</div>
        <div style="color:#94a3b8;font-size:11.5px;margin-bottom:4px;">Group: <span style="color:#e2e8f0;">${group}</span></div>
        <div style="font-size:13px;"><span style="color:#cbd5e1;">Epoch ${epoch}:</span> <b>${val}</b></div>
      `;
      tooltip.style.display = 'block';

      // Position tooltip relative to container
      const containerRect = svg.parentElement.getBoundingClientRect();
      const x = e.clientX - containerRect.left + 14;
      const y = e.clientY - containerRect.top - 20;
      tooltip.style.left = `${Math.min(x, containerRect.width - 240)}px`;
      tooltip.style.top = `${Math.max(10, y)}px`;
    });

    pt.addEventListener('mousemove', (e) => {
      const containerRect = svg.parentElement.getBoundingClientRect();
      const x = e.clientX - containerRect.left + 14;
      const y = e.clientY - containerRect.top - 20;
      tooltip.style.left = `${Math.min(x, containerRect.width - 240)}px`;
      tooltip.style.top = `${Math.max(10, y)}px`;
    });

    pt.addEventListener('mouseleave', () => {
      if (tooltip) tooltip.style.display = 'none';
      resetHighlight();
    });
  });

  // Legend badges hover & click
  allBadges.forEach(b => {
    const sid = b.getAttribute('data-series-id');
    b.addEventListener('mouseenter', () => {
      highlightSeries(sid);
    });
    b.addEventListener('mouseleave', () => {
      resetHighlight();
    });
    b.addEventListener('click', () => {
      if (soloSeriesId === sid) {
        soloSeriesId = null;
        resetHighlight();
      } else {
        soloSeriesId = sid;
        highlightSeries(sid);
      }
    });
  });

  // Filter pills above the chart
  wrapper.querySelectorAll('.pill-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      wrapper.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const grp = btn.getAttribute('data-group');
      soloSeriesId = null;

      allSeries.forEach(s => {
        if (grp === 'all' || s.classList.contains(`group-${grp}`)) {
          s.style.display = '';
        } else {
          s.style.display = 'none';
        }
      });

      wrapper.querySelectorAll('.legend-group').forEach(lg => {
        if (grp === 'all' || lg.classList.contains(`group-${grp}`)) {
          lg.style.display = '';
        } else {
          lg.style.display = 'none';
        }
      });

      resetHighlight();
    });
  });
});

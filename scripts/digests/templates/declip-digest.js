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
    if (soloSeriesId) return;
    allSeries.forEach(s => s.classList.remove('dimmed', 'highlighted'));
    allBadges.forEach(b => b.classList.remove('dimmed', 'active'));
  }

  // Solo on click
  allBadges.forEach(badge => {
    const sid = badge.getAttribute('data-series-id');
    badge.addEventListener('click', (e) => {
      e.stopPropagation();
      if (soloSeriesId === sid) {
        soloSeriesId = null;
        resetHighlight();
      } else {
        soloSeriesId = sid;
        highlightSeries(sid);
      }
    });

    badge.addEventListener('mouseenter', () => highlightSeries(sid));
    badge.addEventListener('mouseleave', () => resetHighlight());
  });

  // Hover on points for tooltip
  svg.querySelectorAll('.series-point').forEach(pt => {
    pt.addEventListener('mouseenter', (e) => {
      const ep = pt.getAttribute('data-epoch');
      const val = pt.getAttribute('data-val');
      const sname = pt.getAttribute('data-series');
      const sid = pt.closest('.chart-series').getAttribute('data-series-id');
      highlightSeries(sid);

      const rect = svg.getBoundingClientRect();
      const ptX = parseFloat(pt.getAttribute('cx'));
      const ptY = parseFloat(pt.getAttribute('cy'));
      const scaleX = rect.width / {W};
      const scaleY = rect.height / {H};

      tooltip.innerHTML = `<strong>${sname}</strong><br>Epoch: ${ep} &bull; Value: <b>${val}</b>`;
      tooltip.style.left = (ptX * scaleX + 12) + 'px';
      tooltip.style.top = (ptY * scaleY - 18) + 'px';
      tooltip.style.display = 'block';
    });

    pt.addEventListener('mouseleave', () => {
      tooltip.style.display = 'none';
      resetHighlight();
    });
  });

  // Filter pills
  wrapper.querySelectorAll('.pill-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const group = btn.getAttribute('data-group');
      btn.parentElement.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      soloSeriesId = null;
      allSeries.forEach(s => {
        if (group === 'all' || s.classList.contains(`group-${group}`)) {
          s.style.display = '';
        } else {
          s.style.display = 'none';
        }
      });

      wrapper.querySelectorAll('.legend-group').forEach(lg => {
        if (group === 'all' || lg.classList.contains(`group-${group}`)) {
          lg.style.display = '';
        } else {
          lg.style.display = 'none';
        }
      });
    });
  });
});

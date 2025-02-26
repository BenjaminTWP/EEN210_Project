// SETUP LIVE MONITOR CHART
var liveChart = new Chart(document.getElementById('liveChart').getContext('2d'), {
    type: 'line',
    data: {
        labels: [],  // Time (MM:SS)
        datasets: [
            { label: 'Acceleration Magnitude', data: [], borderColor: 'red' },
            { label: 'Gyroscope Magnitude', data: [], borderColor: 'blue' }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { title: { display: true, text: "Time (MM:SS)" } },
            y: { title: { display: true, text: "Magnitude" }, ticks: {
                stepSize: 5,  
            } }
        }
    }
});

const MAX_POINTS = 50;

function updateLiveMonitorChart(timestamp, accelMagnitude, gyroMagnitude) {
    liveChart.data.labels.push(timestamp);
    liveChart.data.datasets[0].data.push(accelMagnitude);
    liveChart.data.datasets[1].data.push(gyroMagnitude);

    if (liveChart.data.labels.length > MAX_POINTS) {
        liveChart.data.labels.shift();
        liveChart.data.datasets[0].data.shift();
        liveChart.data.datasets[1].data.shift();
    }

    liveChart.update("none");
}

// SETUP LIVE CHARTS
var liveAccelChart = new Chart(document.getElementById('liveAccelChart').getContext('2d'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            { label: 'Acceleration X', data: [], borderColor: 'red' },
            { label: 'Acceleration Y', data: [], borderColor: 'green' },
            { label: 'Acceleration Z', data: [], borderColor: 'blue' }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false
    }
});

var liveGyroChart = new Chart(document.getElementById('liveGyroChart').getContext('2d'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            { label: 'Gyroscope X', data: [], borderColor: 'purple' },
            { label: 'Gyroscope Y', data: [], borderColor: 'orange' },
            { label: 'Gyroscope Z', data: [], borderColor: 'cyan' }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false
    }
});

const MAX_POINTS = 50; 

function updateLiveCharts(data) {
    liveAccelChart.data.labels.push(""); 
    liveGyroChart.data.labels.push("");

    liveAccelChart.data.datasets[0].data.push(data.acceleration_x);
    liveAccelChart.data.datasets[1].data.push(data.acceleration_y);
    liveAccelChart.data.datasets[2].data.push(data.acceleration_z);

    liveGyroChart.data.datasets[0].data.push(data.gyroscope_x);
    liveGyroChart.data.datasets[1].data.push(data.gyroscope_y);
    liveGyroChart.data.datasets[2].data.push(data.gyroscope_z);

    // Limit entries to MAX_POINTS
    if (liveAccelChart.data.datasets[0].data.length > MAX_POINTS) {
        liveAccelChart.data.labels.shift();
        liveAccelChart.data.datasets.forEach(ds => ds.data.shift());
    }
    if (liveGyroChart.data.datasets[0].data.length > MAX_POINTS) {
        liveGyroChart.data.labels.shift();
        liveGyroChart.data.datasets.forEach(ds => ds.data.shift());
    }

    liveAccelChart.update();
    liveGyroChart.update();
}

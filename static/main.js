
document.getElementById('loading-screen').style.display = 'none'

function uploadData(e) {
  document.getElementById('loading-screen').style.display = 'flex'
  e.preventDefault();
  const fileInput = document.getElementById('csvDataFile');
  const file = fileInput.files[0];

  const formData = new FormData();
  formData.append('file', file);

  axios.post('/upload', formData)
    .then(function (response) {
      console.log(response);
      renderGraphs(response.data.graphs);
      document.getElementById('loading-screen').style.display = 'none'
    })
    .catch(function (error) {
      console.error(error);
    });

  return false;
}

function renderGraphs(names) {
  var container = document.getElementById("graphs-container");

  names.forEach((img) => {
    var graph = document.createElement('img');
    graph.src = `/static/graphs/${img}`;
    graph.alt = 'GRAPH';

    container.appendChild(graph);
  });
}

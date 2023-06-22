
document.getElementById('loading-screen').style.display = 'none'

// function uploadData(e) {
//   document.getElementById('loading-screen').style.display = 'flex'
//   e.preventDefault();
//   const fileInput = document.getElementById('csvDataFile');
//   const file = fileInput.files[0];

//   const formData = new FormData();
//   formData.append('file', file);

//   axios.post('/upload', formData)
//     .then(function (response) {
//       console.log(response);
//       renderGraphs(response.data.graphs);
//       document.getElementById('loading-screen').style.display = 'none'
//     })
//     .catch(function (error) {
//       console.error(error);
//     });

//   return false;
// }

const btns = document.querySelectorAll('button');

btns.forEach((button) => {
  button.addEventListener('click', (e) => {
    document.getElementById('loading-screen').style.display = 'flex'
    axios.get('/upload?modelName=' + button.dataset.name)
      .then(function (res) {
        console.log(res)
        renderGraphs(res.data.graphs)
        document.getElementById('loading-screen').style.display = 'none'
        document.getElementById('date').innerText = res.data.prices.date
        document.getElementById('price-open').innerText = res.data.prices.open
        document.getElementById('price-close').innerText = res.data.prices.adj_close
      })
      .catch(function (res) {
        console.error(res)
      })
  })
})

function renderGraphs(names) {
  var container = document.getElementById("graphs-container");

  names.forEach((img) => {
    var graph = document.createElement('img');
    graph.src = `/static/graphs/${img}`;
    graph.alt = 'GRAPH';

    container.appendChild(graph);
  });
}

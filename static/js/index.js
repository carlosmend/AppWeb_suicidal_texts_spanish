const selector_model = document.getElementById("model");

function sendJSON() {
  const texto = document.getElementById("texto").value;
  const data = { texto };

  fetch("http://localhost:5000//predict", {
    method: "POST",
    headers: { "Content-Type": "application/json; charset=utf-8" },
    body: JSON.stringify(data),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.json();
    })
    .then((data) => {
      console.log(data);
      document.getElementById("resp_all").value = data.moda;
      var list_pred = data.predictions.join("\n");

      document.getElementById("resp_models").value = list_pred.replace(
        /,/g,
        "   "
      );
    })
    .catch((error) => {
      console.error(error);
      document.getElementById("respuesta").value =
        "Error al obtener la respuesta de la API";
    });
}

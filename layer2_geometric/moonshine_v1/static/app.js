
async function addItem(){
  try{
    let pts = JSON.parse(document.getElementById("points").value || "[]");
    let kind = document.getElementById("kind").value || "geom";
    let channel = parseFloat(document.getElementById("channel").value || "3");
    let payload = {kind, points: pts, meta: {channel}};
    let r = await fetch("/api/add", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)});
    let j = await r.json();
    alert("Added: " + JSON.stringify(j));
    loadStats();
  }catch(e){ alert("Bad JSON in points"); }
}
async function doSearch(){
  try{
    let pts = JSON.parse(document.getElementById("qpoints").value || "[]");
    let chart = document.getElementById("chart").value;
    let payload = {points: pts, meta: {}, topk: 10, chart: chart||undefined};
    let r = await fetch("/api/search", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)});
    let j = await r.json();
    document.getElementById("results").textContent = JSON.stringify(j, null, 2);
  }catch(e){ alert("Bad JSON in query points"); }
}
async function loadStats(){
  let r = await fetch("/api/stats"); let j = await r.json();
  document.getElementById("stats").textContent = JSON.stringify(j, null, 2);
}
loadStats();

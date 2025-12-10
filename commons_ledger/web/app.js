function show(id){
  document.querySelectorAll('.pane').forEach(p=>p.style.display='none');
  document.getElementById(id).style.display='block';
}
show('ingest');

function getJSON(id){
  try { return JSON.parse(document.getElementById(id).value); }
  catch(e){ alert('invalid JSON'); return {}; }
}
async function api(path, body){
  const res = await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const json = await res.json();
  alert(JSON.stringify(json,null,2));
  return json;
}
async function post(path, body){
  const res = await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const json = await res.json();
  const id = path.includes('tick')||path.includes('council') ? 'tick_out' :
             path.includes('market') ? 'market_out' :
             path.includes('bounty') ? 'bounty_out' : 'verify_out';
  document.getElementById(id).textContent = JSON.stringify(json,null,2);
  return json;
}
async function get(path){
  const res = await fetch(path);
  const json = await res.json();
  document.getElementById('verify_out').textContent = JSON.stringify(json,null,2);
  return json;
}
async function extractAuto(){
  const src = document.getElementById('feat_src').value;
  const res = await fetch('/features/extract_auto',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:src})});
  const json = await res.json();
  document.getElementById('feat_out').textContent = JSON.stringify(json,null,2);
  document.getElementById('feat_json').value = JSON.stringify(json,null,2);
}
async function validateFeatures(){
  const obj = getJSON('feat_json');
  const res = await fetch('/features/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj)});
  const json = await res.json();
  document.getElementById('feat_out').textContent = JSON.stringify(json,null,2);
}
async function scoreMint(){
  const obj = getJSON('mint_json');
  const res = await fetch('/mint/score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj)});
  const json = await res.json();
  document.getElementById('mint_out').textContent = JSON.stringify(json,null,2);
}
async function mintNow(){
  const obj = getJSON('mint_json');
  const res = await fetch('/mint/mint_now',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj)});
  const json = await res.json();
  document.getElementById('mint_out').textContent = JSON.stringify(json,null,2);
}
async function loadBalance(){
  const res = await fetch('/wallet/balance');
  const json = await res.json();
  document.getElementById('wallet_out').textContent = JSON.stringify(json,null,2);
}

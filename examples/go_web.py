#!/usr/bin/env python3
"""Go web server — fresh GTP process per request."""

import subprocess, os, sys
from http.server import HTTPServer, BaseHTTPRequestHandler

GTP_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'float', 'go_gtp')
CKPT = '/tmp/go_seed.nnckpt'
PORT = 8086

def gtp(commands):
    """Run GTP commands in a fresh process."""
    proc = subprocess.run(
        [GTP_BIN, '--load', CKPT],
        input='\n'.join(commands) + '\nquit\n',
        capture_output=True, text=True, timeout=30
    )
    out = proc.stdout
    results = {}
    current_cmd = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('= '):
            results[current_cmd] = line[2:] if current_cmd else line[2:]
        elif line.startswith('? '):
            results['error'] = line[2:]
    return results

HTML = '''<!DOCTYPE html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Sparse MCTS Go 9x9</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:system-ui,monospace;background:#0b0b14;color:#c8d6e5;display:flex;flex-direction:column;align-items:center;padding:24px;min-height:100vh}
h1{font-size:1rem;letter-spacing:.12em;margin-bottom:4px}.sub{color:#667;font-size:.75rem;margin-bottom:16px}
.g{display:grid;grid-template-columns:repeat(9,42px);grid-template-rows:repeat(9,42px);background:#141428;border:2px solid #2a2a44;border-radius:4px}
.c{width:42px;height:42px;display:flex;align-items:center;justify-content:center;cursor:pointer;position:relative;border:.5px solid #1a1a33}
.c:hover{background:#1e1e3a}.s{width:32px;height:32px;border-radius:50%;position:absolute}
.b{background:radial-gradient(circle at 36% 36%,#555,#000)}.w{background:radial-gradient(circle at 36% 36%,#fff,#aaa)}
.dot{background:radial-gradient(circle at 36% 36%,#667 20%,transparent);width:32px;height:32px;border-radius:50%;position:absolute;pointer-events:none}      
.last{box-shadow:0 0 0 2px #fc3}.info{margin:12px 0;font-size:.8rem;display:flex;gap:16px;align-items:center}
.info span{color:#fc3}.btn{padding:6px 16px;background:#1e1e3a;border:1px solid #334;color:#c8d6e5;cursor:pointer;border-radius:4px;font:inherit;font-size:.75rem}.btn:hover{background:#2a2a55}
#st{color:#fc3;font-size:.8rem;margin:8px 0;min-height:20px}.moves{font-size:.7rem;color:#556;margin-top:8px;max-height:60px;overflow-y:auto}
</style></head><body>
<h1>Sparse MCTS Go 9x9</h1><div class=sub>You Black &middot; Engine White &middot; 100 sims/move &middot; Continuous learning</div>
<div class=info><span id=turn>Your turn</span><button class=btn onclick=reset()>New Game</button></div>
<div id=st></div><div class=g id=board></div><div class=moves id=mvs></div>
<script>
const S=9;let b=Array(S*S).fill(0),t=1,go=false,lm=-1,moves=[];
async function r(){
 let h='';
 for(let y=0;y<S;y++)for(let x=0;x<S;x++){
  let i=y*S+x,v=b[i],isStar=(x===2||x===6)&&(y===2||y===6)||(x===4&&y===4);
  h+='<div class=c onclick="cl('+i+')" id="c'+i+'">'+
   (v?('<div class="s '+(v===1?'b':'w')+(i===lm?' last':'')+'"></div>'):
    (isStar?'<div class=dot></div>':''))+'</div>';
 }
 document.getElementById('board').innerHTML=h;
}
async function cl(i){if(go||t!==1||b[i]!==0)return;p(i,1);await em();}
async function em(){
 t=-1;document.getElementById('turn').textContent='Engine thinking…';
 try{
  let body=moves.join('\\n')+'\\n'+'genmove W';
  let resp=await fetch('/gtp',{method:'POST',body});
  let m=(await resp.text()).trim();
  if(m==='PASS'||!m){go=true;document.getElementById('st').textContent='Game over';}
  else{let x=m.charCodeAt(0)-65,y=parseInt(m.slice(1))-1;if(x>=0&&x<S&&y>=0&&y<S)p(y*S+x,-1);}
 }catch(e){document.getElementById('st').textContent='Error: '+e}
 t=1;document.getElementById('turn').textContent=go?'Game over':'Your turn';
 mv();
}
function p(i,c){b[i]=c;lm=i;let x=i%S,y=Math.floor(i/S);moves.push('play '+(c===1?'B':'W')+' '+String.fromCharCode(65+x)+(y+1));r();}
async function reset(){moves=['boardsize 9','clear_board'];b.fill(0);t=1;go=false;lm=-1;
 document.getElementById('st').textContent='';document.getElementById('turn').textContent='Your turn';r();mv();}
function mv(){document.getElementById('mvs').textContent=moves.slice(2).join(' ')}
r();
</script></body></html>'''

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self._html(HTML)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/gtp':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode()
            commands = body.strip().split('\n')
            try:
                results = gtp(commands)
                answer = results.get('genmove W', 'PASS')
                self._text(answer)
            except Exception as e:
                self._text(f'error: {e}', 500)
        else:
            self.send_error(404)

    def _html(self, body):
        self.send_response(200); self.send_header('Content-Type','text/html'); self.end_headers()
        self.wfile.write(body.encode())
    def _text(self, s, code=200):
        self.send_response(code); self.send_header('Content-Type','text/plain'); self.end_headers()
        self.wfile.write(s.encode())
    def log_message(self, f, *a): pass

print(f'http://127.0.0.1:{PORT}')
HTTPServer(('0.0.0.0', PORT), H).serve_forever()

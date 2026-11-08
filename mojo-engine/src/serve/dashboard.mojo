# dashboard.mojo - the testing chat dashboard served at
# GET /. One static HTML page; EVERY dynamic value on it is
# fetched from the JSON endpoints (/openapi.json is the
# contract it is written against; /health, /v1/models,
# /stats, /history, /leaderboard and /debug/log feed the
# panels). David's house style: panels as tiles, black and
# Army red, no emojis anywhere, the debug log scrolls inside
# its own panel.
#
# Panels: chat (prompt, Send, repeat-run N, reply with token
# counts and timings read from the response JSON), attach
# (image file -> sent in the standard "image" field, the
# endpoint's vision_not_ready refusal is shown as-is;
# document .txt/.md/.zip extracted in the browser with the
# libraries the page carries below - DEFLATE via the
# browser's own DecompressionStream, stored entries read
# directly; .pdf and .7z cannot be unpacked by what the page
# carries and refuse BY NAME in the page, per the lane
# fallback rule), models (fact sheets from /v1/models),
# leaderboard, history, debug log (polls /debug/log every 2
# seconds, pause + auto-scroll toggles, clear view), and
# downloads (transcript and debug log as .txt / .md / .pdf,
# the PDF generated client-side as a minimal valid PDF with
# the full text).
#
# Piece 3 guardrails (amendments 2026-11-08): extracted text
# is prepended with a manifest header (the file list) and
# every file's text carries its own --- FILE: path ---
# header; a token-budget pre-check estimates tokens as
# chars/4 (labeled an estimate on the page) against the
# threshold the server publishes (/health's
# attach_warn_tokens, config MOJO_ENGINE_SERVE_ATTACH_WARN_
# TOKENS, default 4096) and REFUSES to send over-threshold
# context until the operator clicks the explicit override;
# the run record stores the attached file names.

from serve.json import json_escape


def dashboard_html(
    model_name: String, warn_tokens: Int, max_context_bytes: Int
) -> String:
    var out = String("")
    out += "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
    out += "<meta charset=\"utf-8\">\n"
    out += (
        "<meta name=\"viewport\" content=\"width=device-width,"
        " initial-scale=1\">\n"
    )
    out += "<title>mojo-engine test bench</title>\n<style>\n"
    out += (
        "body{background:#060606;color:#f0f0f0;font-family:system-ui,sans-serif;"
        "margin:0;padding:12px;}\n"
    )
    out += "h1{font-size:19px;letter-spacing:2px;margin:4px 0 2px;}\n"
    out += ".sub{color:#bdbdbd;font-size:13px;margin-bottom:12px;}\n"
    out += (
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,"
        "1fr));gap:12px;}\n"
    )
    out += (
        ".tile{background:#141414;border:1px solid #333;border-radius:10px;"
        "padding:12px;}\n"
    )
    out += (
        ".tile h2{font-size:14px;letter-spacing:1px;color:#e8e8e8;margin:0 0 "
        "8px;border-bottom:2px solid #c8102e;padding-bottom:5px;}\n"
    )
    out += (
        "label{display:block;font-size:13px;font-weight:600;margin:8px 0 3px;}"
        "\n"
    )
    out += (
        "textarea,input[type=text],input[type=number]{width:100%;box-sizing:"
        "border-box;background:#0b0b0b;color:#f0f0f0;border:1px solid #444;"
        "border-radius:6px;padding:8px;font-size:14px;}\n"
    )
    out += (
        "button{background:#c8102e;color:#fff;border:0;border-radius:6px;"
        "padding:8px 13px;font-size:14px;font-weight:600;cursor:pointer;"
        "margin:6px 6px 0 0;}\n"
    )
    out += "button.alt{background:#2a2a2a;border:1px solid #555;}\n"
    out += "button:disabled{opacity:.45;cursor:default;}\n"
    out += ".mono{font-family:ui-monospace,monospace;font-size:12px;}\n"
    out += (
        ".scroll{overflow-y:auto;max-height:300px;white-space:pre-wrap;"
        "word-break:break-word;}\n"
    )
    out += ".warn{color:#ffb3ba;font-size:13px;}\n"
    out += ".ok{color:#9fe8a8;font-size:13px;}\n"
    out += "table{border-collapse:collapse;width:100%;font-size:12px;}\n"
    out += "td,th{border-bottom:1px solid #333;padding:4px 5px;text-align:left;}\n"
    out += ".reply{background:#0b0b0b;border:1px solid #333;border-radius:6px;padding:8px;white-space:pre-wrap;word-break:break-word;}\n"
    out += "</style>\n</head>\n<body>\n"
    out += "<h1>MOJO-ENGINE TEST BENCH</h1>\n"
    out += (
        "<div class=\"sub\">Model: <span id=\"modelName\">"
        + json_escape(model_name)
        + "</span> &middot; <span id=\"healthLine\">loading health</span></div>\n"
    )
    out += "<div class=\"grid\">\n"

    # -- chat tile ------------------------------------------------------------
    out += "<section class=\"tile\">\n<h2>CHAT</h2>\n"
    out += "<label for=\"prompt\">Prompt</label>\n"
    out += (
        "<textarea id=\"prompt\" rows=\"3\">The capital of France"
        " is</textarea>\n"
    )
    out += "<label for=\"reps\">Repeat runs</label>\n"
    out += "<input type=\"number\" id=\"reps\" value=\"1\" min=\"1\" max=\"20\">\n"
    out += "<label for=\"maxtok\">Max tokens</label>\n"
    out += "<input type=\"number\" id=\"maxtok\" value=\"64\" min=\"1\" max=\"512\">\n"
    out += (
        "<div><button id=\"sendBtn\">Send</button>"
        "<button id=\"attachBtn\" class=\"alt\">Attach file</button>"
        "<input type=\"file\" id=\"fileInput\" hidden></div>\n"
    )
    out += "<div id=\"attachInfo\" class=\"mono\"></div>\n"
    out += "<div id=\"attachWarn\" class=\"warn\"></div>\n"
    out += (
        "<div><button id=\"overrideBtn\" class=\"alt\" hidden>Send oversized"
        " context anyway</button></div>\n"
    )
    out += "<label>Reply</label>\n<div class=\"reply\" id=\"reply\">(none yet)</div>\n"
    out += "<div class=\"mono\" id=\"timings\"></div>\n"
    out += "</section>\n"

    # -- debug tile -----------------------------------------------------------
    out += "<section class=\"tile\">\n<h2>DEBUG LOG</h2>\n"
    out += (
        "<div class=\"scroll mono\" id=\"debugLog\" style=\"max-height:330px\">"
        "</div>\n"
    )
    out += (
        "<div><button id=\"pauseBtn\" class=\"alt\">Pause</button>"
        "<button id=\"scrollBtn\" class=\"alt\">Auto-scroll: on</button>"
        "<button id=\"clearBtn\" class=\"alt\">Clear view</button></div>\n"
    )
    out += "<label>Download log</label>\n<div>"
    out += (
        "<button class=\"alt\" data-dl=\"log:txt\">TXT</button>"
        "<button class=\"alt\" data-dl=\"log:md\">MD</button>"
        "<button class=\"alt\" data-dl=\"log:pdf\">PDF</button></div>\n"
    )
    out += "<label>Download transcript</label>\n<div>"
    out += (
        "<button class=\"alt\" data-dl=\"chat:txt\">TXT</button>"
        "<button class=\"alt\" data-dl=\"chat:md\">MD</button>"
        "<button class=\"alt\" data-dl=\"chat:pdf\">PDF</button></div>\n"
    )
    out += "</section>\n"

    # -- models tile ----------------------------------------------------------
    out += "<section class=\"tile\">\n<h2>MODELS</h2>\n"
    out += "<div id=\"models\" class=\"mono scroll\" style=\"max-height:380px\"></div>\n"
    out += "</section>\n"

    # -- leaderboard tile -----------------------------------------------------
    out += "<section class=\"tile\">\n<h2>LEADERBOARD</h2>\n"
    out += "<div id=\"leaderboard\"></div>\n</section>\n"

    # -- history tile ---------------------------------------------------------
    out += "<section class=\"tile\">\n<h2>RUN HISTORY</h2>\n"
    out += (
        "<label for=\"histSource\">Source filter</label>\n"
        "<input type=\"text\" id=\"histSource\" placeholder=\"dashboard, cli or"
        " mcp\">\n"
    )
    out += "<div><button id=\"histBtn\" class=\"alt\">Refresh history</button>"
    out += "<button id=\"exportBtn\" class=\"alt\">Export run bundle</button></div>\n"
    out += "<div id=\"history\" class=\"mono scroll\" style=\"max-height:330px\"></div>\n"
    out += "</section>\n"

    # -- stats tile -----------------------------------------------------------
    out += "<section class=\"tile\">\n<h2>STATS</h2>\n"
    out += "<div id=\"stats\" class=\"mono\"></div>\n</section>\n"
    out += "</div>\n"

    # -- script ---------------------------------------------------------------
    out += "<script>\n"
    out += "\"use strict\";\n"
    out += "var WARN_TOKENS=" + String(warn_tokens) + ";\n"
    out += "var MAX_CONTEXT_BYTES=" + String(max_context_bytes) + ";\n"
    out += _dashboard_js()
    out += "</script>\n</body>\n</html>\n"
    return out


def _dashboard_js() -> String:
    var js = String("")
    js += "var attached={kind:null,name:\"\",names:[],text:\"\",image:null};\n"
    js += "var transcript=[];\nvar debugLines=[];\nvar debugPaused=false;\nvar debugAutoScroll=true;\n"
    js += "function $(id){return document.getElementById(id);}\n"
    js += "function esc(s){return String(s).replace(/&/g,\"&amp;\").replace(/</g,\"&lt;\");}\n"
    js += "function fmtTs(sec){if(!sec){return \"-\";}try{return new Date(sec*1000).toLocaleString();}catch(e){return String(sec);}}\n"

    # health / stats / models / leaderboard / history loaders
    js += "async function loadHealth(){try{var r=await fetch(\"/health\");var h=await r.json();"
    js += "$(\"healthLine\").textContent=h.engine+\" \"+h.version+\" \\u00b7 backend \"+h.backend+\" \\u00b7 sampler \"+(h.sampler&&h.sampler.name?h.sampler.name:h.sampler)+\" \\u00b7 model loaded: \"+h.model_loaded;"
    js += "if(h.attach_warn_tokens){WARN_TOKENS=h.attach_warn_tokens;}}catch(e){$(\"healthLine\").textContent=\"health unavailable: \"+e;}}\n"
    js += "async function loadStats(){try{var r=await fetch(\"/stats\");var s=await r.json();"
    js += "var html=\"\";for(var k in s){html+=esc(k)+\": \"+esc(s[k])+\"<br>\";}$(\"stats\").innerHTML=html;}catch(e){}}\n"
    js += "async function loadModels(){try{var r=await fetch(\"/v1/models\");var m=await r.json();var html=\"\";"
    js += "for(var i=0;i<m.models.length;i++){var s=m.models[i];if(s.error){html+=\"<b>\"+esc(s.file_name||s.path)+\"</b>: \"+esc(s.error)+\"<br><br>\";continue;}"
    js += "var L=s.layers||{};var V=s.vision||{};"
    js += "html+=\"<b>\"+esc(s.id)+\"</b>\"+(s.served?\" (served)\":\"\")+\"<br>arch \"+esc(s.architecture)+\" \\u00b7 layers \"+L.total+\" (attn \"+L.attention+\", ssm \"+L.ssm+\", sliding \"+L.sliding+\", global \"+L.global+\")<br>ctx \"+s.context_length+\" \\u00b7 hidden \"+s.embedding_length+\" \\u00b7 heads \"+s.head_count+\"/\"+s.head_count_kv+\" \\u00b7 vocab \"+s.vocab_size+\"<br>file \"+s.file_bytes+\" B \\u00b7 tensors \"+s.tensor_count+\" \\u00b7 vision: \"+(V.present?(\"yes (\"+esc(V.source)+\", \"+V.tensor_count+\" tensors)\"):\"no\")+\"<br><br>\";}"
    js += "$(\"models\").innerHTML=html||\"(no models)\";}catch(e){$(\"models\").textContent=\"models unavailable: \"+e;}}\n"
    js += "async function loadBoard(){try{var r=await fetch(\"/leaderboard\");var b=await r.json();"
    js += "var html=\"<table><tr><th>Model</th><th>Hardware</th><th>Kind</th><th>Decode tok/s</th><th>Prefill tok/s</th><th>Runs</th></tr>\";"
    js += "for(var i=0;i<b.rows.length;i++){var row=b.rows[i];html+=\"<tr><td>\"+esc(row.model)+\"</td><td>\"+esc(row.backend_target)+\"</td><td>\"+esc(row.kind)+\"</td><td>\"+(row.best_decode_tok_s||\"-\")+\"</td><td>\"+(row.best_prefill_tok_s||\"-\")+\"</td><td>\"+row.runs+\"</td></tr>\";}"
    js += "html+=\"</table>\";$(\"leaderboard\").innerHTML=html;}catch(e){$(\"leaderboard\").textContent=\"leaderboard unavailable: \"+e;}}\n"
    js += "async function loadHistory(){try{var q=$(\"histSource\").value?(\"?source=\"+encodeURIComponent($(\"histSource\").value)):\"\";"
    js += "var r=await fetch(\"/history\"+q);var h=await r.json();var html=\"\";"
    js += "for(var i=h.runs.length-1;i>=0;i--){var run=h.runs[i];html+=fmtTs(run.timestamp)+\" \\u00b7 \"+esc(run.source)+(run.imported?\" (imported)\":\"\")+\" \\u00b7 \"+esc(run.model)+\"<br>prompt \"+run.prompt_tokens+\" (bucket \"+run.prompt_bucket+\") \\u00b7 prefill \"+run.prefill_ms+\" ms / \"+Number(run.prefill_tok_s).toFixed(1)+\" tok/s \\u00b7 decode \"+Number(run.decode_tok_s).toFixed(2)+\" tok/s \\u00b7 out \"+run.tokens_generated+(run.quality_verdict?(\" \\u00b7 \"+esc(run.quality_verdict)):\"\")+\"<br><br>\";}"
    js += "$(\"history\").innerHTML=html||\"(no runs yet)\";}catch(e){$(\"history\").textContent=\"history unavailable: \"+e;}}\n"

    # chat send + repeat
    js += "function estimateTokens(text){return Math.ceil(text.length/4);}\n"
    js += "function updateAttach(){var info=\"\",warn=\"\";$(\"overrideBtn\").hidden=true;"
    js += "if(attached.kind===\"text\"){var est=estimateTokens(attached.text);info=\"Attached: \"+attached.names.join(\", \")+\" \\u00b7 \"+attached.text.length+\" chars \\u00b7 estimated \"+est+\" tokens (estimate: chars/4)\";"
    js += "if(est>WARN_TOKENS){warn=\"Estimated context is over the \"+WARN_TOKENS+\" token budget pre-check. Sending is refused until you override.\";$(\"overrideBtn\").hidden=false;}}"
    js += "else if(attached.kind===\"image\"){info=\"Attached image: \"+attached.name+\" (the engine refuses image input by name until a vision path lands)\";}"
    js += "$(\"attachInfo\").textContent=info;$(\"attachWarn\").textContent=warn;}\n"
    js += "async function sendChat(allowOversize){var prompt=$(\"prompt\").value;var reps=Math.max(1,Math.min(20,parseInt($(\"reps\").value)||1));"
    js += "var maxTok=parseInt($(\"maxtok\").value)||64;var btn=$(\"sendBtn\");btn.disabled=true;"
    js += "if(attached.kind===\"text\"&&!allowOversize&&estimateTokens(attached.text)>WARN_TOKENS){$(\"reply\").textContent=\"Refused by the token budget pre-check: estimated \"+estimateTokens(attached.text)+\" tokens over the \"+WARN_TOKENS+\" cap. Click the override to send anyway.\";btn.disabled=false;return;}"
    js += "for(var rep=0;rep<reps;rep++){var payload={prompt:prompt,max_tokens:maxTok,source:\"dashboard\",temperature:0.0};"
    js += "if(attached.kind===\"text\"){payload.context_text=attached.text;payload.attachments=attached.names;}"
    js += "if(attached.kind===\"image\"){payload.image=attached.image;}"
    js += "try{var r=await fetch(\"/v1/chat/completions\",{method:\"POST\",headers:{\"Content-Type\":\"application/json\"},body:JSON.stringify(payload)});"
    js += "var data=await r.json();"
    js += "if(data.error){$(\"reply\").textContent=\"Refusal: \"+(data.error.code||\"error\")+\": \"+(data.error.message||\"\");transcript.push({role:\"system\",text:\"refusal \"+(data.error.code||\"\")+\" for prompt: \"+prompt});}"
    js += "else{var content=data.choices[0].message.content;$(\"reply\").textContent=content;var t=data.timings||{};var u=data.usage||{};"
    js += "$(\"timings\").innerHTML=\"run \"+(rep+1)+\"/\"+reps+\" \\u00b7 prompt tokens \"+u.prompt_tokens+\" \\u00b7 completion tokens \"+u.completion_tokens+\"<br>prefill \"+t.prefill_ms+\" ms (\"+Number(t.prefill_tok_s).toFixed(1)+\" tok/s) \\u00b7 decode \"+Number(t.decode_tok_s).toFixed(2)+\" tok/s \\u00b7 total \"+t.total_ms+\" ms \\u00b7 ttft \"+(t.ttft_ms===null?\"n/a (non-streaming)\":t.ttft_ms)+\" \\u00b7 sampler greedy temp \"+data.sampler.temperature;"
    js += "transcript.push({role:\"user\",text:prompt});transcript.push({role:\"assistant\",text:content});}"
    js += "}catch(e){$(\"reply\").textContent=\"Request failed: \"+e;}}"
    js += "btn.disabled=false;loadStats();loadHistory();loadBoard();}\n"
    js += "$(\"sendBtn\").addEventListener(\"click\",function(){sendChat(false);});\n"
    js += "$(\"overrideBtn\").addEventListener(\"click\",function(){sendChat(true);});\n"

    # attach handling: zip (stored + deflate via DecompressionStream), txt/md
    js += "function manifestHeader(names){var h=\"Attached files (\"+names.length+\"):\";for(var i=0;i<names.length;i++){h+=\"\\n- \"+names[i];}return h+\"\\n\";}\n"
    js += "async function inflateRaw(bytes){var ds=new DecompressionStream(\"deflate-raw\");var stream=new Blob([bytes]).stream().pipeThrough(ds);var buf=await new Response(stream).arrayBuffer();return new Uint8Array(buf);}\n"
    js += "async function readZip(file){var buf=new Uint8Array(await file.arrayBuffer());var view=new DataView(buf.buffer,buf.byteOffset,buf.byteLength);"
    js += "var files=[];var i=0;while(i+30<=buf.length&&view.getUint32(i,true)===0x04034b50){var method=view.getUint16(i+8,true);var compSize=view.getUint32(i+18,true);var nameLen=view.getUint16(i+26,true);var extraLen=view.getUint16(i+28,true);"
    js += "var name=new TextDecoder().decode(buf.subarray(i+30,i+30+nameLen));var dataStart=i+30+nameLen+extraLen;var dataEnd=dataStart+compSize;"
    js += "if(compSize===0&&name.endsWith(\"/\")){i=dataEnd;continue;}"
    js += "var raw=buf.subarray(dataStart,dataEnd);var text=null;"
    js += "if(method===0){text=new TextDecoder().decode(raw);}else if(method===8){try{text=new TextDecoder().decode(await inflateRaw(raw));}catch(e){text=null;}}"
    js += "if(text!==null&&name.length){files.push({name:name,text:text});}i=dataEnd;}"
    js += "if(!files.length&&buf.length>=22){var eocd=-1;for(var j=buf.length-22;j>=0;j--){if(view.getUint32(j,true)===0x06054b50){eocd=j;break;}}"
    js += "if(eocd>=0){var count=view.getUint16(eocd+10,true);var cdOff=view.getUint32(eocd+16,true);var k=cdOff;"
    js += "for(var n=0;n<count&&k+46<=buf.length&&view.getUint32(k,true)===0x02014b50;n++){var m2=view.getUint16(k+10,true);var cs=view.getUint32(k+20,true);var nl=view.getUint16(k+28,true);var el=view.getUint16(k+30,true);var cl=view.getUint16(k+32,true);var lh=view.getUint32(k+42,true);"
    js += "var nm=new TextDecoder().decode(buf.subarray(k+46,k+46+nl));var lnl=view.getUint16(lh+26,true);var lel=view.getUint16(lh+28,true);var ds2=lh+30+lnl+lel;var raw2=buf.subarray(ds2,ds2+cs);var tx=null;"
    js += "if(m2===0){tx=new TextDecoder().decode(raw2);}else if(m2===8){try{tx=new TextDecoder().decode(await inflateRaw(raw2));}catch(e){tx=null;}}"
    js += "if(tx!==null&&nm.length){files.push({name:nm,text:tx});}k+=46+nl+el+cl;}}"
    js += "return files;}\n"
    js += "function refuseInPage(msg){attached={kind:null,name:\"\",names:[],text:\"\",image:null};$(\"attachInfo\").textContent=\"\";$(\"attachWarn\").textContent=msg;}\n"
    js += "$(\"attachBtn\").addEventListener(\"click\",function(){$(\"fileInput\").click();});\n"
    js += "$(\"fileInput\").addEventListener(\"change\",async function(){var file=this.files[0];this.value=\"\";if(!file){return;}var name=file.name;var lower=name.toLowerCase();"
    js += "if(/\\.(png|jpe?g|gif|webp|bmp)$/.test(lower)){var rd=new FileReader();rd.onload=function(){attached={kind:\"image\",name:name,names:[name],text:\"\",image:rd.result};updateAttach();};rd.readAsDataURL(file);return;}"
    js += "if(lower.endsWith(\".pdf\")){refuseInPage(\"serve_attachment_unpack: PDF text extraction is not carried by this page in v1; paste the text or attach a .txt/.md/.zip instead.\");return;}"
    js += "if(lower.endsWith(\".7z\")){refuseInPage(\"serve_attachment_unpack: 7z unpacking is not carried by this page in v1; attach a .zip or plain text instead.\");return;}"
    js += "try{if(lower.endsWith(\".zip\")){var entries=await readZip(file);if(!entries.length){refuseInPage(\"serve_attachment_unpack: no readable text entries found in \"+name);return;}"
    js += "var names=[];var combined=manifestHeader(entries.map(function(e){return e.name;}));for(var i2=0;i2<entries.length;i2++){names.push(entries[i2].name);combined+=\"\\n--- FILE: \"+entries[i2].name+\" ---\\n\"+entries[i2].text;}"
    js += "if(combined.length>MAX_CONTEXT_BYTES){refuseInPage(\"serve_context_too_large: extracted text is \"+combined.length+\" bytes, server cap is \"+MAX_CONTEXT_BYTES);return;}"
    js += "attached={kind:\"text\",name:name,names:names,text:combined,image:null};updateAttach();return;}"
    js += "if(/\\.(txt|md)$/.test(lower)){var text=await file.text();var combined2=manifestHeader([name])+\"\\n--- FILE: \"+name+\" ---\\n\"+text;"
    js += "if(combined2.length>MAX_CONTEXT_BYTES){refuseInPage(\"serve_context_too_large: extracted text is \"+combined2.length+\" bytes, server cap is \"+MAX_CONTEXT_BYTES);return;}"
    js += "attached={kind:\"text\",name:name,names:[name],text:combined2,image:null};updateAttach();return;}"
    js += "refuseInPage(\"serve_attachment_type: only images, .txt, .md, .pdf, .zip or .7z can be attached; got \"+name);"
    js += "}catch(e){refuseInPage(\"serve_attachment_unpack: could not read \"+name+\": \"+e);}});\n"

    # debug polling + downloads
    js += "async function pollDebug(){if(debugPaused){return;}try{var r=await fetch(\"/debug/log\");var d=await r.json();debugLines=d.lines||[];"
    js += "var el=$(\"debugLog\");el.textContent=debugLines.join(\"\\n\");if(debugAutoScroll){el.scrollTop=el.scrollHeight;}}catch(e){}}\n"
    js += "setInterval(pollDebug,2000);\n"
    js += "$(\"pauseBtn\").addEventListener(\"click\",function(){debugPaused=!debugPaused;this.textContent=debugPaused?\"Resume\":\"Pause\";if(!debugPaused){pollDebug();}});\n"
    js += "$(\"scrollBtn\").addEventListener(\"click\",function(){debugAutoScroll=!debugAutoScroll;this.textContent=\"Auto-scroll: \"+(debugAutoScroll?\"on\":\"off\");});\n"
    js += "$(\"clearBtn\").addEventListener(\"click\",function(){$(\"debugLog\").textContent=\"\";});\n"
    js += "function saveBlob(name,content,type){var b=new Blob([content],{type:type});var a=document.createElement(\"a\");a.href=URL.createObjectURL(b);a.download=name;a.click();setTimeout(function(){URL.revokeObjectURL(a.href);},4000);}\n"
    js += "function pdfEscape(s){return s.replace(/\\\\/g,\"\\\\\\\\\").replace(/\\(/g,\"\\\\(\").replace(/\\)/g,\"\\\\)\");}\n"
    js += "function makePdf(text){var lines=[];var raw=text.split(\"\\n\");for(var i=0;i<raw.length;i++){var s=raw[i];while(s.length>95){lines.push(s.slice(0,95));s=s.slice(95);}lines.push(s);}"
    js += "var perPage=48;var pages=[];for(var p=0;p<lines.length;p+=perPage){pages.push(lines.slice(p,p+perPage));}if(!pages.length){pages.push([\"\"]);}"
    js += "var objs=[];objs.push(\"<< /Type /Catalog /Pages 2 0 R >>\");var kids=pages.map(function(_,i){return (3+i*2)+\" 0 R\";}).join(\" \");"
    js += "objs.push(\"<< /Type /Pages /Kids [\"+kids+\"] /Count \"+pages.length+\" >>\");"
    js += "for(var i2=0;i2<pages.length;i2++){objs.push(\"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents \"+(4+i2*2)+\" 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Courier >> >> >> >>\");"
    js += "var content=\"BT /F1 9 Tf 14 TL 40 750 Td\";for(var j=0;j<pages[i2].length;j++){content+=\" (\"+pdfEscape(pages[i2][j])+\") Tj T*\";}content+=\" ET\";"
    js += "objs.push(\"<< /Length \"+content.length+\" >>\\nstream\\n\"+content+\"\\nendstream\");}"
    js += "var pdf=\"%PDF-1.4\\n\";var offsets=[];for(var k=0;k<objs.length;k++){offsets.push(pdf.length);pdf+=(k+1)+\" 0 obj\\n\"+objs[k]+\"\\nendobj\\n\";}"
    js += "var xref=pdf.length;pdf+=\"xref\\n0 \"+(objs.length+1)+\"\\n0000000000 65535 f \\n\";for(var k2=0;k2<offsets.length;k2++){pdf+=String(offsets[k2]).padStart(10,\"0\")+\" 00000 n \\n\";}"
    js += "pdf+=\"trailer\\n<< /Size \"+(objs.length+1)+\" /Root 1 0 R >>\\nstartxref\\n\"+xref+\"\\n%%EOF\";return pdf;}\n"
    js += "function transcriptText(){var out=\"mojo-engine test bench transcript\\n\";for(var i=0;i<transcript.length;i++){out+=transcript[i].role+\": \"+transcript[i].text+\"\\n\";}return out;}\n"
    js += "document.querySelectorAll(\"[data-dl]\").forEach(function(btn){btn.addEventListener(\"click\",function(){var parts=btn.getAttribute(\"data-dl\").split(\":\");var which=parts[0];var format=parts[1];"
    js += "var text=which===\"log\"?(debugLines.join(\"\\n\")||$(\"debugLog\").textContent):transcriptText();var base=which===\"log\"?\"debug-log\":\"transcript\";"
    js += "if(format===\"pdf\"){saveBlob(base+\".pdf\",makePdf(text),\"application/pdf\");}else{saveBlob(base+\".\"+format,text,\"text/plain\");}});});\n"
    js += "$(\"histBtn\").addEventListener(\"click\",loadHistory);\n"
    js += "$(\"exportBtn\").addEventListener(\"click\",async function(){try{var r=await fetch(\"/history/export\");var t=await r.text();saveBlob(\"run-bundle.json\",t,\"application/json\");}catch(e){}});\n"
    js += "loadHealth();loadStats();loadModels();loadBoard();loadHistory();pollDebug();\n"
    return js

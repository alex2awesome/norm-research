<!-- URL: https://www.energy.gov/cmei/communicationstandards/style-guide-full-text -->
<!-- HTTP: 200 SIZE:   386950 -->
<!DOCTYPE html>
<html lang="en" dir="ltr" prefix="content: http://purl.org/rss/1.0/modules/content/  dc: http://purl.org/dc/terms/  foaf: http://xmlns.com/foaf/0.1/  og: http://ogp.me/ns#  rdfs: http://www.w3.org/2000/01/rdf-schema#  schema: http://schema.org/  sioc: http://rdfs.org/sioc/ns#  sioct: http://rdfs.org/sioc/types#  skos: http://www.w3.org/2004/02/skos/core#  xsd: http://www.w3.org/2001/XMLSchema# ">

  <head>
    <meta charset="utf-8" />
<script type="text/javascript">/*
 Copyright and licenses see https://www.dynatrace.com/company/trust-center/customers/reports/ */
(function(){function Xa(){var Aa;(Aa=void 0===Ya.dialogArguments&&navigator.cookieEnabled)||(document.cookie="__dTCookie=1;SameSite=Lax",Aa=document.cookie.includes("__dTCookie"),Aa=(document.cookie="__dTCookie=1; expires=Thu, 01-Jan-1970 00:00:01 GMT",Aa));return Aa}function Ia(){if(Xa()){var Aa=Ya.dT_,bb=null==Aa?void 0:Aa.platformPrefix,Ua;if(Ua=!Aa||bb){var Da;Ua=(null===(Da=document.currentScript)||void 0===Da?void 0:Da.getAttribute("data-dtconfig"))||"domain=energy.gov|reportUrl=/rb_33bb55c9-1410-4de1-8b54-d592dfdc60af|app=cff6532c80af1c10|owasp=1|featureHash=A7NVfgqrux|xb=^bs/sitewide_alert^bs/load^p^bs/sitewide^bs_alert^bs/load|rdnt=1|uxrgce=1|cuc=dgtcdnr2|mel=100000|expw=1|dpvc=1|lastModification=1776197503156|postfix=dgtcdnr2|tp=500,50,0|srbbv=2|agentUri=/ruxitagentjs_A7NVfgqrux_10333260303165926.js|auto=1|domain=energy.gov|rid=RID_-1330322208|rpid=1010679433";
Ua=(Da=null==Aa?void 0:Aa.minAgentVersion)&&"10333260303165926"<`${Da}`?(console.log("[CookiePrefix/initConfig] Min agent version detected, and javascript agent is older - the javascript agent will not be initialized!"),!1):bb&&!Ua.includes(`cuc=${bb}`)?(console.log(`[CookiePrefix/initConfig] value of platformPrefix [${bb}] does not match the "cuc" value in the current config - initConfig will not initialize!`),!1):!0}Ua&&(Aa={cfg:"domain=energy.gov|reportUrl=/rb_33bb55c9-1410-4de1-8b54-d592dfdc60af|app=cff6532c80af1c10|owasp=1|featureHash=A7NVfgqrux|xb=^bs/sitewide_alert^bs/load^p^bs/sitewide^bs_alert^bs/load|rdnt=1|uxrgce=1|cuc=dgtcdnr2|mel=100000|expw=1|dpvc=1|lastModification=1776197503156|postfix=dgtcdnr2|tp=500,50,0|srbbv=2|agentUri=/ruxitagentjs_A7NVfgqrux_10333260303165926.js|auto=1|domain=energy.gov|rid=RID_-1330322208|rpid=1010679433",
iCE:Xa,platformCookieOffset:null==Aa?void 0:Aa.platformCookieOffset},bb&&(Aa.platformPrefix=bb),Ya.dT_=Aa)}}function Ta(Aa){return"function"==typeof Aa}function db(){let Aa=window.performance;return Aa&&Fa&&Pa&&!!Aa.timeOrigin&&Ta(Aa.now)&&Ta(Aa.getEntriesByType)&&Ta(Aa.getEntriesByName)&&Ta(Aa.setResourceTimingBufferSize)&&Ta(Aa.clearResourceTimings)}this.dT_&&dT_.prm&&dT_.prm();var Ya="undefined"!=typeof globalThis?globalThis:"undefined"!=typeof window?window:void 0,Pa="function"==typeof PerformanceResourceTiming||
"object"==typeof PerformanceResourceTiming,Fa="function"==typeof PerformanceNavigationTiming||"object"==typeof PerformanceNavigationTiming;if((!("documentMode"in document)||Number.isNaN(document.documentMode))&&db()){let Aa=Ya.dT_;!Aa||!Aa.cfg&&Aa.platformPrefix?Ia():Aa.gCP||(Ya.console.log("Duplicate agent injection detected, turning off redundant initConfig."),Aa.di=1)}})();
(function(){function Xa(){let f=0;try{f=Math.round(Ma.performance.timeOrigin)}catch(p){}if(0>=f||Number.isNaN(f)||!Number.isFinite(f))ke.push({severity:"Warning",type:"ptoi",text:`performance.timeOrigin is invalid, with a value of [${f}]. Falling back to performance.timing.navigationStart`}),f=Me;return ae=f,jd=Ia,ae}function Ia(){return ae}function Ta(){return jd()}function db(){let f=0;try{f=Math.round(Ma.performance.now())}catch(p){}return 0>=f||Number.isNaN(f)||!Number.isFinite(f)?Date.now()-
jd():f}function Ya(f,p){if(!p)return"";var y=f+"=";f=p.indexOf(y);if(0>f)return"";for(;0<=f;){if(0===f||" "===p.charAt(f-1)||";"===p.charAt(f-1))return y=f+y.length,f=p.indexOf(";",f),0<=f?p.substring(y,f):p.substring(y);f=p.indexOf(y,f+y.length)}return""}function Pa(f){return Ya(f,document.cookie)}function Fa(){}function Aa(f,p=[]){var y;if(!f||"object"!=typeof f&&"function"!=typeof f)return!1;let Q="number"!=typeof p?p:[],V=null,va=[];switch("number"==typeof p?p:5){case 0:V="Array";break;case 1:V=
"Boolean";break;case 2:V="Number";break;case 3:V="String";break;case 4:V="Function";break;case 5:V="Object";break;case 6:V="Date";va.push("getTime");break;case 7:V="Error";va.push("name","message");break;case 8:V="Element";break;case 9:V="HTMLElement";break;case 10:V="HTMLImageElement";va.push("complete");break;case 11:V="PerformanceEntry";break;case 13:V="PerformanceResourceTiming";break;case 14:V="PerformanceNavigationTiming";break;case 15:V="CSSRule";va.push("cssText","parentStyleSheet");break;
case 16:V="CSSStyleSheet";va.push("cssRules","insertRule");break;case 17:V="Request";va.push("url");break;case 18:V="Response";va.push("ok","status","statusText");break;case 19:V="Set";va.push("add","entries","forEach");break;case 20:V="Map";va.push("set","entries","forEach");break;case 21:V="Worker";va.push("addEventListener","postMessage","terminate");break;case 22:V="XMLHttpRequest";va.push("open","send","setRequestHeader");break;case 23:V="SVGScriptElement";va.push("ownerSVGElement","type");break;
case 24:V="HTMLMetaElement";va.push("httpEquiv","content","name");break;case 25:V="HTMLHeadElement";break;case 26:V="ArrayBuffer";break;case 27:V="ShadowRoot",va.push("host","mode")}let {Qe:Za,bf:Lb}={Qe:V,bf:va};if(!Za)return!1;p=Lb.length?Lb:Q;if(!Q.length)try{if(Ma[Za]&&f instanceof Ma[Za]||Object.prototype.toString.call(f)==="[object "+Za+"]")return!0;if(f&&f.nodeType&&1===f.nodeType){let Pb=null===(y=f.ownerDocument.defaultView)||void 0===y?void 0:y[Za];if("function"==typeof Pb&&f instanceof
Pb)return!0}}catch(Pb){}for(let Pb of p)if("string"!=typeof Pb&&"number"!=typeof Pb&&"symbol"!=typeof Pb||!(Pb in f))return!1;return!!p.length}function bb(f,p,y,Q){"undefined"==typeof Q&&(Q=Da(p,!0));"boolean"==typeof Q&&(Q=Da(p,Q));f===Ma?le&&le(p,y,Q):Xf&&Aa(f,21)?ai.call(f,p,y,Q):f.addEventListener&&(f===Ma.document||f===Ma.document.documentElement?kh.call(f,p,y,Q):le.call(f,p,y,Q));Q=!1;for(let V=kd.length;0<=--V;){let va=kd[V];if(va.object===f&&va.event===p&&va.Z===y){Q=!0;break}}Q||kd.push({object:f,
event:p,Z:y})}function Ua(f,p,y,Q){for(let V=kd.length;0<=--V;){let va=kd[V];if(va.object===f&&va.event===p&&va.Z===y){kd.splice(V,1);break}}"undefined"==typeof Q&&(Q=Da(p,!0));"boolean"==typeof Q&&(Q=Da(p,Q));f===Ma?pf&&pf(p,y,Q):f.removeEventListener&&(f===Ma.document||f===Ma.document.documentElement?lh.call(f,p,y,Q):pf.call(f,p,y,Q))}function Da(f,p){let y=!1;try{if(le&&me.includes(f)){let Q=Object.defineProperty({},"passive",{get(){y=!0}});le("test",Fa,Q)}}catch(Q){}return y?{passive:!0,capture:p}:
p}function ta(){let f=kd;for(let p=f.length;0<=--p;){let y=f[p];Ua(y.object,y.event,y.Z)}kd=[]}function na(f){return"function"==typeof f}function wa(f){if(f&&void 0!==f[qf])return f[qf];let p=na(f)&&/{\s+\[native code]/.test(Function.prototype.toString.call(f));return f&&(f[qf]=p),p}function ua(f,p,...y){return void 0!==Function.prototype.bind&&wa(Function.prototype.bind)?Function.prototype.bind.call(f,p,...y):function(...Q){return f.apply(p,(y||[]).concat(Array.prototype.slice.call(arguments)||[]))}}
function xa(){if(Yf){let f=new Yf;if(Ze)for(let p of Sg)void 0!==Ze[p]&&(f[p]=ua(Ze[p],f));return f}return new Ma.XMLHttpRequest}function Ca(){return yc()}function eb(f,p,...y){function Q(){delete Hd[V];f.apply(this,arguments)}let V,va=mh;return"apply"in va?(y.unshift(Q,p),V=va.apply(Ma,y)):V=va(Q,p),Hd[V]=!0,V}function G(f){delete Hd[f];let p=Lf;"apply"in p?p.call(Ma,f):p(f)}function J(f){Td.push(f)}function Y(f){let p=Td.length;for(;p--;)if(Td[p]===f){Td.splice(p,1);break}}function qa(){return Td}
function O(f,p){return te(f,p)}function Z(f){be(f)}function ha(f,p){if(!rf||!$e)return"";f=new rf([f],{type:p});return $e(f)}function T(f,p){return Zf?new Zf(f,p):void 0}function C(f){"function"==typeof f&&wg.push(f)}function W(){return wg}function ia(){return Me}function R(f){function p(...y){if("number"!=typeof y[0]||!Hd[y[0]])try{return f.apply(this,y)}catch(Q){return f(y[0])}}return p[Qd]=f,p}function ja(){return ke}function pa(){jd=Xa;yc=()=>Math.round(jd()+db());(Number.isNaN(yc())||0>=yc()||
!Number.isFinite(yc()))&&(yc=()=>Date.now())}function Va(){ue&&(Ma.clearTimeout=Lf,Ma.clearInterval=be,ue=!1)}function ib(f){f=encodeURIComponent(f);let p=[];if(f)for(let y=0;y<f.length;y++){let Q=f.charAt(y);p.push($f[Q]||Q)}return p.join("")}function $a(f){return f.includes("^")&&(f=f.split("^^").join("^"),f=f.split("^dq").join('"'),f=f.split("^rb").join(">"),f=f.split("^lb").join("<"),f=f.split("^p").join("|"),f=f.split("^e").join("="),f=f.split("^s").join(";"),f=f.split("^c").join(","),f=f.split("^bs").join("\\")),
f}function zb(){return ve}function Db(f){ve=f}function Tb(f){let p=La("rid"),y=La("rpid");p&&(f.rid=p);y&&(f.rpid=y)}function $b(f){if(f=f.xb){f=$a(f);try{ve=new RegExp(f,"i")}catch(p){}}else ve=void 0}function dc(f){return"n"===f||"s"===f||"l"===f?`;SameSite=${xg[f]}`:""}function uc(f,p,y){let Q=1,V=0;do document.cookie=f+'=""'+(p?";domain="+p:"")+";path="+y.substring(0,Q)+"; expires=Thu, 01 Jan 1970 00:00:01 GMT;",Q=y.indexOf("/",Q),V++;while(-1!==Q&&5>V)}function U(f,p=document.cookie){return Ya(f,
p)}function gb(f){var p;return function(){if(p)return p;var y,Q,V,va,Za=-1!==we?we:(null===(Q=null===(y=Ma.dT_)||void 0===y?void 0:y.scv)||void 0===Q?void 0:Q.call(y,"postfix"))||(null===(va=null===(V=Ma.dT_)||void 0===V?void 0:V.gCP)||void 0===va?void 0:va.call(V));y=(y=(we=Za,Za))?"".concat(f).concat(y):f;return p=y,y}}function Ic(){let f=(document.domain||location.hostname||"").split(".");return 1>=f.length?[]:f}function ob(f,p){p=`dTValidationCookieValue;path=/;domain=${f}${dc(p)}`;return document.cookie=
`${yg()}=${p}`,U(yg())?(uc(yg(),f,"/"),!0):!1}function nc(){var f=nh.dT_.scv("agentUri");if(!f)return"ruxitagentjs";f=f.substring(f.lastIndexOf("/")+1);return f.includes("_")?f.substring(0,f.indexOf("_")):f}function Sb(){return Ma.dT_}function Ab(f){fa("eao")?Mf=f:Ma.console.log(`${xe} App override configuration is not enabled! The provided app config callback method will not be invoked!`)}function ac(f){let p=navigator.userAgent;if(!p)return null;let y=p.indexOf("DTHybrid/1.0");return-1===y?null:
(f=p.substring(y).match(new RegExp(`(^|\\s)${f}=([^\\s]+)`)))?f[2]:null}function S(){return navigator.userAgent.includes("Googlebot")||navigator.userAgent.includes("AdsBot-Google")}function ba(f){let p=f,y=Math.pow(2,32);return function(){return p=(1664525*p+1013904223)%y,p/y}}function za(f,p){return Number.isNaN(f)||Number.isNaN(p)?Math.floor(33*Ud()):Math.floor(Ud()*(p-f+1))+f}function ea(f){if(Number.isNaN(f)||0>f)return"";var p=Ma.crypto;if(p&&!S())p=p.getRandomValues(new Uint8Array(f));else{p=
[];for(var y=0;y<f;y++)p.push(za(0,32))}f=Array(f);for(y=0;y<p.length;y++){let Q=Math.abs(p[y]%32);f.push(String.fromCharCode(Q+(9>=Q?48:55)))}return f.join("")}function wb(f=!0){Vd=f}function Wa(){return Bd}function hb(){let f=Ma.dT_;try{let V=Ma.parent;for(;V&&V!==Ma;){let va=V.dT_;var p;if(p=!!(va&&"ea"in va)&&"10333260303165926"===va.version){var y=f,Q=va;let Za=!y.bcv("ac")||y.scv("app")===Q.scv("app"),Lb=y.scv("postfix"),Pb=Q.scv("postfix");p=Za&&Lb===Pb}if(p)return va.tdto();V=V!==V.parent?
V.parent:void 0}}catch(V){}return f}function jb(f,p="",y){var Q=f.indexOf("h"),V=f.indexOf("v"),va=f.indexOf("e");let Za=f.substring(f.indexOf("$")+1,Q);Q=-1!==V?f.substring(Q+1,V):f.substring(Q+1);!p&&-1!==V&&(-1!==va?p=f.substring(V+1,va):p=f.substring(V+1));f=null;y||(V=Number.parseInt(Za.split("_")[0]),va=yc()%ye,y=(va<V&&(va+=ye),V+9E5>va));return y&&(f={frameId:Za,actionId:"-"===Q?"-":Number.parseInt(Q),visitId:""}),{visitId:p,Ed:f}}function Jb(f,p,y){var Q=Ea("pcl");Q=f.length-Q;0<Q&&f.splice(0,
Q);Q=Jd(U(Ed(),y));let V=[],va=Q?`${Q}$`:"";for(Za of f)"-"!==Za.actionId&&V.push(`${va}${Za.frameId}h${Za.actionId}`);f=V.join("p");var Za=(f||(Vd&&(Qb(!0,"a",y),wb(!1)),f+=`${Q}$${Bd}h-`),f+=`v${p||L(y)}`,f+="e0",f);Mc(ze(),Za,void 0,fa("ssc"))}function ec(f,p=document.cookie){var y=U(ze(),p);p=[];if(y&&"-"!==y){var Q=y.split("p");y="";for(let V of Q)Q=jb(V,y,f),y=Q.visitId,Q.Ed&&p.push(Q.Ed);for(let V of p)V.visitId=y}return p}function zc(f,p){let y=document.cookie;p=ec(p,y);let Q=!1;for(let V of p)V.frameId===
Bd&&(V.actionId=f,Q=!0);Q||p.push({frameId:Bd,actionId:f,visitId:""});Jb(p,void 0,y)}function lc(){return ne}function pb(f,p){try{Ma.localStorage.setItem(f,p)}catch(y){}}function pc(f){try{Ma.localStorage.removeItem(f)}catch(p){}}function sb(f){try{return Ma.localStorage.getItem(f)}catch(p){}return null}function x(f,p=document.cookie||""){return p.split(f+"=").length-1}function u(f,p){var y=x(f,p);if(1<y){p=La("domain")||Ma.location.hostname;let Q=Ma.location.hostname,V=Ma.location.pathname,va=0,
Za=0;Xc.add(f);do{let Lb=Q.substring(va);if(Lb!==p||"/"!==V){uc(f,Lb===p?"":Lb,V);let Pb=x(f);Pb<y&&(Xc.add(Lb),y=Pb)}va=Q.indexOf(".",va)+1;Za++}while(0!==va&&10>Za&&1<y);La("domain")&&1<y&&uc(f,"",V)}}function r(){var f=document.cookie;u(ze(),f);u(Ed(),f);u(nd(),f);f=Xc.size;0<f&&f!==ce&&(ce=f,ke.push({severity:"Error",type:"dcn",text:`Duplicate cookie name${1!==f?"s":""} detected: ${Array.from(Xc).join(",")}`}))}function D(){Xc=new Set;r();J(function(f,p,y,Q){0===Xc.size||p||(f.av(Q,"dCN",Array.from(Xc).join(",")),
Xc.clear(),ce=0)})}function L(f){return N(f)||Qb(!0,"c",f)}function N(f){if(P(f)<=yc())return Qb(!0,"t",f);let p=Cb(f);if(p){let y=zg.exec(p);f=y&&3===y.length&&32===y[1].length&&!Number.isNaN(Number.parseInt(y[2]))?(Xb(dd(),p),p):Qb(!0,"i",f)}else f=Qb(!0,"c",f);return f}function ca(f,p){var y=yc();p=Sc(p).mc;f&&(p=y);ld(y+Rd+"|"+p);Zc()}function oa(f){let p="t"+(yc()-P(f));Qb(!0,p,f)}function Cb(f){var p,y;return null!==(y=null===(p=ec(!0,f)[0])||void 0===p?void 0:p.visitId)&&void 0!==y?y:Cc(dd())}
function Fb(){let f=ea(af);try{f=f.replace(/\d/g,function(p){p=.1*Number.parseInt(p);return String.fromCharCode(Math.floor(25*p+65))})}catch(p){throw Aa(p,7),p;}return f+="-0",f}function mc(f,p){let y=ec(!1,p);Jb(y,f,p);Xb(dd(),f);ca(!0)}function tc(f,p,y){return Qb(f,p,y)}function Qb(f,p,y){f&&(Ne=!0);f=Cb(y);y=Fb();mc(y);if(Cb(document.cookie))for(let Q of Oe)Q(y,Ne,p,f);return y}function Bc(f){Oe.push(f)}function Cd(f){f=Oe.indexOf(f);-1!==f&&Oe.splice(f)}function Zc(f){sf&&G(sf);sf=eb(ed,P(f)-
yc())}function ed(){let f=document.cookie;return P(f)<=yc()?(mb(oa,[f]),!0):(K(Zc),!1)}function ld(f){let p=ua(Mc,null,nd(),f,void 0,fa("ssc"));p();let y=U(nd());""!==y&&f!==y&&(r(),p(),f===U(nd())||ag||(ke.push({severity:"Error",type:"dcn",text:"Could not sanitize cookies"}),ag=!0));Xb(nd(),f)}function od(f,p){p=U(f,p);return p||(p=Cc(f)||""),p}function pd(){var f=N()||"";Xb(dd(),f);f=od(nd());ld(f);pc(dd());pc(nd())}function Sc(f){let p={ad:0,mc:0};if(f=od(nd(),f))try{let y=f.split("|");2===y.length&&
(p.ad=Number.parseInt(y[0]),p.mc=Number.parseInt(y[1]))}catch(y){}return p}function P(f){f=Sc(f);return Math.min(f.ad,f.mc+Ag)}function aa(f){Rd=f}function F(){let f=Ne;return Ne=!1,f}function ra(){ed()||ca(!1)}function Ga(){if(fa("nsfnv")){var f=U(Ed());if(!f.includes(`${oe}-`)){let p=ya(f).serverId;f=f.replace(`${oe}${p}`,`${oe}${`${-1*za(2,Pe)}`.replace("-","-2D")}`);$c(f)}}}function Ra(){K(function(){if(!yd()){let f=-1*za(2,Pe),p=ea(af);$c(`v_4${oe}${`${f}`.replace("-","-2D")}_sn_${p}`)}},!0);
Bc(Ga)}function ya(f){var p;let y={},Q={sessionId:"",serverId:"",overloadState:0,appState:y};var V=f.split(bf);if(2<V.length&&0===V.length%2){f=+V[1];if(Number.isNaN(f)||3>f)return Q;f={};for(var va=2;va<V.length;va++){let Za=V[va];Za.startsWith(Tg)?y[Za.substring(6).toLowerCase()]=+V[va+1]:f[Za]=V[va+1];va++}f.sn?(V=f.sn,Q.sessionId=V.length===af||12>=V.length?V:""):Q.sessionId="hybrid";if(f.srv){a:{V=f.srv.replace("-2D","-");if(!Number.isNaN(+V)&&(va=Number.parseInt(V),va>=-Pe&&va<=Pe))break a;
V=""}Q.serverId=V}V=+f.ol;1===V&&lb(vc());0<=V&&2>=V&&(Q.overloadState=V);f=+f.prv;Number.isNaN(f)||(Q.privacyState=1>f||4<f?1:f);f=null===(p=La("app"))||void 0===p?void 0:p.toLowerCase();p=y[f];!Number.isNaN(p)&&0===p&&lb(vc())}return Q}function lb(f){let p=Sb();f||(p.disabled=!0,p.overloadPrevention=!0)}function Na(){if(yb()||Ma.dynatraceMobile){let f=Pa("dtAdkSettings")||ac("dtAdkSettings")||"";return ya(f).privacyState||null}return null}function yb(){var f,p;let y=Ma.MobileAgent;if(y)return y;
let Q=null===(p=null===(f=Ma.webkit)||void 0===f?void 0:f.messageHandlers)||void 0===p?void 0:p.MobileAgent;if(Q)return{incrementActionCount:()=>Q.postMessage({api:"incrementActionCount"})}}function mb(f,p){return!Yb()||Sb().overloadPrevention&&!vc()?null:f.apply(this,p)}function Yb(){let f=Na();return 2===f||1===f?!1:!fa("coo")||fa("cooO")||vc()}function cc(f,p){try{Ma.sessionStorage.setItem(f,p)}catch(y){}}function Xb(f,p){mb(cc,[f,p])}function Cc(f){try{return Ma.sessionStorage.getItem(f)}catch(p){}return null}
function Dc(f){try{Ma.sessionStorage.removeItem(f)}catch(p){}}function qc(f){document.cookie=f+'="";path=/'+(La("domain")?";domain="+La("domain"):"")+"; expires=Thu, 01 Jan 1970 00:00:01 GMT;"}function Ec(f,p,y,Q){p||0===p?(p=(p+"").replace(/[\n\r;]/g,"_"),f=f+"="+p+";path=/"+(La("domain")?";domain="+La("domain"):""),y&&(f+=";expires="+y.toUTCString()),f+=dc(La("cssm")),Q&&"https:"===location.protocol&&(f+=";Secure"),document.cookie=f):qc(f)}function Mc(f,p,y,Q){mb(Ec,[f,p,y,Q])}function qd(f){return!!f.includes("v_4")}
function Nc(f){f=U(Ed(),f);f||(f=(f=Cc(Ed()))&&qd(f)?($c(f),f):"");return qd(f)?f:""}function $c(f){Mc(Ed(),f,void 0,fa("ssc"))}function Ae(f){return(f=f||Nc())?ya(f):{sessionId:"",serverId:"",overloadState:0,appState:{}}}function Jd(f){return Ae(f).serverId}function yd(f){return Ae(f).sessionId}function vc(){return navigator.userAgent.includes(Bg)}function zd(f){let p={};f=f.split("|");for(let y of f)f=y.split("="),2===f.length&&(p[f[0]]=decodeURIComponent(f[1].replace(/\+/g," ")));return p}function A(){let f=
nc();return(f.indexOf("dbg")===f.length-3?f.substring(0,f.length-3):f)+"_"+La("app")+"_Store"}function B(f,p={},y){f=f.split("|");for(var Q of f){f=Q;let V=Q.indexOf("="),va;-1===V?(va=f,p[va]="1"):(f=Q.substring(0,V),va=f,p[va]=Q.substring(V+1,Q.length))}!y&&(y=p.spc)&&(Q=document.createElement("textarea"),Q.innerHTML=y,p.spc=Q.value);return p}function I(f){var p;return null!==(p=ic[f])&&void 0!==p?p:de[f]}function fa(f){f=I(f);return"false"===f||"0"===f?!1:!!f}function Ea(f){var p=I(f);p=Number.parseInt(p);
return Number.isNaN(p)&&(p=de[f]),p}function La(f){if("app"===f){a:{try{if(Mf){var p=Mf();if(p&&"string"==typeof p&&bg.test(p)){var y=p;break a}Ma.console.log(`${xe} Registered app config callback method did not return a string in the correct app id format ${bg} The default app ID will be used instead! Got value: ${p}`)}}catch(Q){Ma.console.log(`${xe} Failed to execute the registered app config callback method! The default app ID will be used instead!`,Q)}y=void 0}if(y)return y}return(I(f)||"")+""}
function fb(f,p){ic[f]=p+""}function Bb(f){return ic=f,ic}function Ib(f){let p=ic[f];p&&(ic[f]=p.includes("#"+f.toUpperCase())?"":p)}function Kb(f){var p=f.agentUri;null!=p&&p.includes("_")&&(p=/([A-Za-z]*)\d{0,4}_(\w*)_\d+/g.exec(p))&&p.length&&2<p.length&&(f.featureHash=p[2])}function bc(f){var p=Sb().platformCookieOffset;if("number"==typeof p){a:{var y=La("cssm");var Q=Ic();if(Q.length)for(p=Q.slice(p);p.length;){if(ob(p.join("."),y)){y=p.join(".");break a}p.shift()}y=""}y&&(f.domain=y)}else if(y=
f.domain||"",p=(p=location.hostname)&&y?p===y||p.includes("."+y,p.length-("."+y).length):!0,!y||!p){f.domainOverride||(f.domainOriginal=f.domain||"",f.domainOverride=`${location.hostname},${y}`,delete f.domain);a:{Q=La("cssm");let va=Ic().reverse();if(va.length){var V=va[0];for(let Za=1;Za<=va.length;Za++){if(ob(V,Q)){Q=V;break a}let Lb=va[Za];Lb&&(V=`${Lb}.${V}`)}}Q=""}Q&&(f.domain=Q);p||ke.push({type:"dpi",severity:"Warning",text:`Configured domain "${y}" is invalid for current location "${location.hostname}". Agent will use "${f.domain}" instead.`})}}
function m(f,p){bc(f);let y=ic.pVO;(y&&(f.pVO=y),p)||(f.bp=(f.bp||de.bp)+"")}function v(f,p){if(null==f||!f.attributes)return p;let y=p;var Q=f.attributes.getNamedItem("data-dtconfig");if(!Q)return y;f=f.src;Q=Q.value;let V={};ic.legacy="1";var va=/([A-Za-z]*)_(\w*)_(\d+)/g;let Za=null;va=(f&&(Za=va.exec(f),null!=Za&&Za.length&&(V.agentUri=f,V.featureHash=Za[2],V.agentLocation=f.substring(0,f.indexOf(Za[1])-1),V.buildNumber=Za[3])),Q&&B(Q,V,!0),bc(V),V);if(!p)y=va;else if(!va.syntheticConfig)return va;
return y}function t(){return ic}function E(f){return de[f]===I(f)}function K(f,p=!1){Yb()?f():p?Cg.push(f):Dg.push(f)}function ka(){if(!fa("coo")||Yb())return!1;for(let f of Dg)eb(f,0);Dg=[];fb("cooO",!0);for(let f of Cg)f();return Cg.length=0,!0}function la(){if(!fa("coo")||!Yb())return!1;fb("cooO",!1);qc(Ed());qc(ze());qc(oh());qc("dtAdk");qc(ad());qc(nd());try{Dc(dd()),Dc(nd()),pc(dd()),pc(nd()),Dc(ad()),Dc(Ed()),pc(A()),pc("dtAdk")}catch(f){}return!0}function Ja(){var f=U(ad());if(!f||45!==(null==
f?void 0:f.length))if(f=sb(ad())||Cc(ad()),45!==(null==f?void 0:f.length)){ph=!0;let p=yc()+"";f=(p+=ea(45-p.length),p)}return Sa(f),f}function Sa(f){let p;if(fa("dpvc")||fa("pVO"))Xb(ad(),f);else{p=new Date;var y=p.getMonth()+Math.min(24,Math.max(1,Ea("rvcl")));p.setMonth(y);y=ad();mb(pb,[y,f])}Mc(ad(),f,p,fa("ssc"))}function cb(){return ph}function nb(f){let p=U(ad());qc(ad());Dc(ad());pc(ad());fb("pVO",!0);Sa(p);f&&mb(pb,["dt-pVO","1"]);pd()}function Eb(){pc("dt-pVO");fa("pVO")&&(fb("pVO",!1),
Ja());Dc(ad());pd()}function oc(f,p,y,Q,V,va){let Za=document.createElement("script");Za.setAttribute("src",f);p&&Za.setAttribute("defer","defer");y&&(Za.onload=y);Q&&(Za.onerror=Q);V&&Za.setAttribute("id",V);if(f=va){var Lb;f=null!=(null===(Lb=document.currentScript)||void 0===Lb?void 0:Lb.attributes.getNamedItem("integrity"))}f&&Za.setAttribute("integrity",va);Za.setAttribute("crossorigin","anonymous");va=document.getElementsByTagName("script")[0];va.parentElement.insertBefore(Za,va)}function rc(f){return Nf+
"/"+ee+"_"+f+"_"+(Ea("buildNumber")||Sb().version)+".js"}function Fc(){}function Oc(f,p){return function(){f.apply(p,arguments)}}function Ub(f){if(!(this instanceof Ub))throw new TypeError("Promises must be constructed via new");if("function"!=typeof f)throw new TypeError("not a function");this.pa=0;this.Qb=!1;this.ra=void 0;this.Ea=[];Ug(f,this)}function bd(f,p){for(;3===f.pa;)f=f.ra;0===f.pa?f.Ea.push(p):(f.Qb=!0,Ub.sb(function(){var y=1===f.pa?p.Ib:p.Jb;if(null===y)(1===f.pa?Fd:cf)(p.promise,f.ra);
else{try{var Q=y(f.ra)}catch(V){cf(p.promise,V);return}Fd(p.promise,Q)}}))}function Fd(f,p){try{if(p===f)throw new TypeError("A promise cannot be resolved with itself.");if(p&&("object"==typeof p||"function"==typeof p)){var y=p.then;if(p instanceof Ub){f.pa=3;f.ra=p;tf(f);return}if("function"==typeof y){Ug(Oc(y,p),f);return}}f.pa=1;f.ra=p;tf(f)}catch(Q){cf(f,Q)}}function cf(f,p){f.pa=2;f.ra=p;tf(f)}function tf(f){2===f.pa&&0===f.Ea.length&&Ub.sb(function(){f.Qb||Ub.Wb(f.ra)});for(var p=0,y=f.Ea.length;p<
y;p++)bd(f,f.Ea[p]);f.Ea=null}function Mi(f,p,y){this.Ib="function"==typeof f?f:null;this.Jb="function"==typeof p?p:null;this.promise=y}function Ug(f,p){var y=!1;try{f(function(Q){y||(y=!0,Fd(p,Q))},function(Q){y||(y=!0,cf(p,Q))})}catch(Q){y||(y=!0,cf(p,Q))}}function qh(){return df.sb=function(f){if("string"==typeof f)throw Error("Promise polyfill called _immediateFn with string");f()},df.Wb=()=>{},df}function cg(){var f;(f=void 0===Ma.dialogArguments&&navigator.cookieEnabled)||(document.cookie="__dTCookie=1;SameSite=Lax",
f=document.cookie.includes("__dTCookie"),f=(document.cookie="__dTCookie=1; expires=Thu, 01-Jan-1970 00:00:01 GMT",f));return f}function rh(){var f,p;try{null===(p=null===(f=yb())||void 0===f?void 0:f.incrementActionCount)||void 0===p||p.call(f)}catch(y){}}function dg(){let f=Ma.dT_;Ma.dT_={di:0,version:"10333260303165926",cfg:f?f.cfg:"",iCE:f?cg:function(){return navigator.cookieEnabled},ica:1,disabled:!1,disabledReason:"",overloadPrevention:!1,gAST:ia,ww:T,stu:ha,nw:Ca,st:eb,si:O,aBPSL:J,rBPSL:Y,
gBPSL:qa,aBPSCC:C,gBPSCC:W,buildType:"dynatrace",gSSV:Cc,sSSV:Xb,rSSV:Dc,rvl:pc,iVSC:qd,p3SC:ya,dC:qc,sC:Mc,esc:ib,gSId:Jd,gDtc:yd,gSC:Nc,sSC:$c,gC:Pa,cRN:za,cRS:ea,cfgO:t,pCfg:zd,pCSAA:B,cFHFAU:Kb,sCD:m,bcv:fa,ncv:Ea,scv:La,stcv:fb,rplC:Bb,cLSCK:A,gFId:Wa,gBAU:rc,iS:oc,eWE:K,oEIEWA:mb,eA:ka,dA:la,iNV:cb,gVID:Ja,dPV:nb,ePV:Eb,sVIdUP:wb,sVTT:aa,sVID:mc,rVID:N,gVI:L,gNVIdN:tc,gARnVF:F,cAUV:ra,uVT:ca,aNVL:Bc,rNVL:Cd,gPC:ec,cPC:zc,sPC:Jb,clB:Va,ct:G,aRI:Tb,iXB:$b,gXBR:zb,sXBR:Db,de:$a,iEC:rh,rnw:db,gto:Ta,
ael:bb,rel:Ua,sup:Da,cuel:ta,iAEPOO:Yb,iSM:vc,gxwp:xa,iIO:Aa,prm:qh,cI:Z,gidi:ja,iDCV:E,gCF:U,gPSMB:Na,lvl:sb,gCP:null!=f&&f.platformPrefix?function(){return f.platformPrefix||""}:void 0,platformCookieOffset:null==f?void 0:f.platformCookieOffset,tdto:hb,gFLAI:lc,rACC:Ab,gAN:nc,gAFUA:ac}}function eg(){qh();var f;Yf=Ma.XMLHttpRequest;var p=null===(f=Ma.XMLHttpRequest)||void 0===f?void 0:f.prototype;if(p){Ze={};for(let Q of Sg)void 0!==p[Q]&&(Ze[Q]=p[Q])}le=Ma.addEventListener;pf=Ma.removeEventListener;
kh=Ma.document.addEventListener;lh=Ma.document.removeEventListener;mh=Ma.setTimeout;te=Ma.setInterval;ue||(Lf=Ma.clearTimeout,be=Ma.clearInterval);f=cg?cg():navigator.cookieEnabled;var y;p=U("dtAdkSettings")||(null===(y=uf)||void 0===y?void 0:y.getItem("dtAdkSettings"))||ac("dtAdkSettings")||"";y=1===(ya(p).overloadState||null);if(f){if(y)return!1}else return!1;return!0}function vf(...f){Ma.console.log.apply(Ma.console,f)}this.dT_&&dT_.prm&&dT_.prm();var nh="undefined"!=typeof window?window:self,
Ma="undefined"!=typeof globalThis?globalThis:"undefined"!=typeof window?window:void 0,Lf,be,mh,te,ae,jd,Qd=Symbol.for("dt_2");Symbol.for("dt_3");var le,pf,kh,lh,Xf=Ma.Worker,ai=Xf&&Xf.prototype.addEventListener,kd=[],me=["touchstart","touchend","scroll"],qf=Symbol.for("dt2-isNativeFunction"),Yf,Sg="abort getAllResponseHeaders getResponseHeader open overrideMimeType send setRequestHeader".split(" "),Ze,Zf=Ma.Worker,rf=Ma.Blob,$e=Ma.URL&&Ma.URL.createObjectURL,ue=!1,Td,wg=[],ke=[],Me,uf,Hd={},yc,$f=
{"!":"%21","~":"%7E","*":"%2A","(":"%28",")":"%29","'":"%27",ig:"%24",";":"%3B",",":"%2C"},ve,xg={l:"Lax",s:"Strict",n:"None"},we=-1,ze=gb("dtPC"),Ed=gb("dtCookie"),nd=gb("rxvt"),ad=gb("rxVisitor"),yg=gb("dTValidationCookie"),oh=gb("dtSa");gb("dtsrVID");var de,xe="[Dynatrace RUM JavaScript]",bg=/^[\da-f]{12,16}$/,Mf,Ud,ef,Vd=!1,Bd,ye=6E8,ne,af=32,dd=gb("rxvisitid"),Xc,ce=0,zg=/([A-Z]+)-(\d+)/,Oe=[],Rd,Ag,Ne=!1,sf,ag=!1,bf="_",Pe=32768,oe=`${bf}srv${bf}`,Tg="app-3A",Bg="RuxitSynthetic",Sd=/c(\w+)=([^ ]+(?: (?!c?\w+=)[^ ()c]+)*)/g,
ic={},Dg=[],Cg=[],ph=!1,sh,Nf,ee,Eg=setTimeout;Ub.prototype.catch=function(f){return this.then(null,f)};Ub.prototype.then=function(f,p){var y=new this.constructor(Fc);return bd(this,new Mi(f,p,y)),y};Ub.prototype.finally=function(f){var p=this.constructor;return this.then(function(y){return p.resolve(f()).then(function(){return y})},function(y){return p.resolve(f()).then(function(){return p.reject(y)})})};Ub.all=function(f){return new Ub(function(p,y){function Q(Lb,Pb){try{if(Pb&&("object"==typeof Pb||
"function"==typeof Pb)){var fd=Pb.then;if("function"==typeof fd){fd.call(Pb,function(Kd){Q(Lb,Kd)},y);return}}V[Lb]=Pb;0===--va&&p(V)}catch(Kd){y(Kd)}}if(!f||"undefined"==typeof f.length)return y(new TypeError("Promise.all accepts an array"));var V=Array.prototype.slice.call(f);if(0===V.length)return p([]);for(var va=V.length,Za=0;Za<V.length;Za++)Q(Za,V[Za])})};Ub.allSettled=function(f){return new this(function(p,y){function Q(Za,Lb){if(Lb&&("object"==typeof Lb||"function"==typeof Lb)){var Pb=Lb.then;
if("function"==typeof Pb){Pb.call(Lb,function(fd){Q(Za,fd)},function(fd){V[Za]={status:"rejected",reason:fd};0===--va&&p(V)});return}}V[Za]={status:"fulfilled",value:Lb};0===--va&&p(V)}if(!f||"undefined"==typeof f.length)return y(new TypeError(typeof f+" "+f+" is not iterable(cannot read property Symbol(Symbol.iterator))"));var V=Array.prototype.slice.call(f);if(0===V.length)return p([]);var va=V.length;for(y=0;y<V.length;y++)Q(y,V[y])})};Ub.resolve=function(f){return f&&"object"==typeof f&&f.constructor===
Ub?f:new Ub(function(p){p(f)})};Ub.reject=function(f){return new Ub(function(p,y){y(f)})};Ub.race=function(f){return new Ub(function(p,y){if(!f||"undefined"==typeof f.length)return y(new TypeError("Promise.race accepts an array"));for(var Q=0,V=f.length;Q<V;Q++)Ub.resolve(f[Q]).then(p,y)})};Ub.sb="function"==typeof setImmediate&&function(f){setImmediate(f)}||function(f){Eg(f,0)};Ub.Wb=function(f){"undefined"!=typeof console&&console&&console.warn("Possible Unhandled Promise Rejection:",f)};var df=
Ub,Ni="function"==typeof PerformanceResourceTiming||"object"==typeof PerformanceResourceTiming,th="function"==typeof PerformanceNavigationTiming||"object"==typeof PerformanceNavigationTiming;(function(){var f,p,y,Q;if(Q=!("documentMode"in document&&!Number.isNaN(document.documentMode)))Q=(Q=window.performance)&&th&&Ni&&!!Q.timeOrigin&&na(Q.now)&&na(Q.getEntriesByType)&&na(Q.getEntriesByName)&&na(Q.setResourceTimingBufferSize)&&na(Q.clearResourceTimings);if(Q)if(Q=!(null!==(f=navigator.userAgent)&&
void 0!==f&&f.includes("RuxitSynthetic")),!Ma.dT_||!Ma.dT_.cfg||"string"!=typeof Ma.dT_.cfg||"initialized"in Ma.dT_&&Ma.dT_.initialized)null!==(p=Ma.dT_)&&void 0!==p&&p.gCP?vf("[CookiePrefix/initCode] initCode with cookie prefix already initialized, not initializing initCode!"):vf("InitConfig not found or agent already initialized! This is an injection issue."),Ma.dT_&&(Ma.dT_.di=3);else if(Q)try{dg();de={ade:"",aew:!0,apn:"",agentLocation:"",agentUri:"",app:"",ac:!0,async:!1,ase:!1,auto:!1,bp:3,
bs:!1,buildNumber:0,csprv:!0,cepl:16E3,cls:!0,ccNcss:!1,coo:!1,cooO:!1,cssm:"0",cwtDomains:"",cwtUrl:"w2dax4xgw3",cwt:"",cors:!1,cuc:"",cce:!1,cux:!1,postfix:"",dataDtConfig:"",debugName:"",dvl:500,dASXH:!1,disableCookieManager:!1,dKAH:!1,disableLogging:!1,dmo:!1,doel:!1,dpch:!1,dpvc:!1,disableXhrFailures:!1,domain:"",domainOverride:"",domainOriginal:"",doNotDetect:"",ds:!0,dsndb:!1,dsa:!1,dsss:!1,dssv:!0,earxa:!0,eao:!1,exp:!1,eni:!0,epacod:!1,etc:!0,expw:!1,instr:"",evl:"",fa:!1,fvdi:!1,featureHash:"",
hvt:216E5,imm:!1,ign:"",iub:"",iqvn:!1,initializedModules:"",lastModification:0,lupr:!0,lab:!1,legacy:!1,lt:!0,mb:"",md:"",mdp:"",mdl:"",mcepsl:100,mdn:5E3,mhl:4E3,mpl:1024,mmds:2E4,msl:3E4,bismepl:2E3,mel:200,mepp:10,moa:30,mrt:3,ntd:!1,nsfnv:!1,ncw:!1,oat:180,ote:!1,owasp:!1,pcl:20,pt:!0,perfbv:1,prfSmpl:0,pVO:!1,peti:!1,raxeh:!0,rdnt:0,nosr:!0,reportUrl:"dynaTraceMonitor",rid:"",ridPath:"",rpid:"",rcdec:12096E5,rtl:0,rtp:2,rtt:1E3,rtu:200,rvcl:24,sl:100,ssc:!1,svNB:!1,srad:!0,srbbv:1,srbw:!0,srcst:!1,
srdinitrec:!0,srmr:100,srms:"1,1,,,",srsr:1E5,srtbv:3,srtd:1,srtr:500,srvr:"",srvi:0,srwo:!1,srre:"",srxcss:!0,srxicss:!0,srficros:!0,srmrc:!1,srsdom:!0,srfirrbv:!0,srcss:!0,srmcrl:1,srmcrv:10,st:3E3,spc:"",syntheticConfig:!1,tal:0,tt:100,tvc:3E3,uxdce:!1,uxdcw:1500,uxrgce:!0,uxrgcm:"100,25,300,3;100,25,300,3",uam:!1,uana:"data-dtname,data-dtName",uanpi:0,pui:!1,usrvd:!0,vrt:!1,vcfi:!0,vcsb:!1,vcit:1E3,vct:50,vcx:50,vscl:0,vncm:1,vrei:"",xb:"",chw:"",xt:0,xhb:""};if(!eg())throw Error("Error during initCode initialization");
try{uf=Ma.localStorage}catch(ud){}ke=[];pa();Me=yc();Td=[];Hd={};ue||(Ma.clearTimeout=R(Lf),Ma.clearInterval=R(be),ue=!0);let Wd=Math.random(),vd=Math.random();ef=0!==Wd&&0!==vd&&Wd!==vd;if(S()){var V=performance.getEntriesByType("navigation")[0];if(V){f=0;for(var va in V){let ud=+V[va];!Number.isNaN(ud)&&Number.isFinite(ud)&&(f+=Math.abs(ud))}var Za=Math.floor(1E4*f)}else Za=yc();Ud=ba(Za)}else ef?Ud=Math.random:Ud=ba(yc());Bd=Me%ye+"_"+Number.parseInt(za(0,1E3)+"");a:{let ud=Sb().cfg;ic={reportUrl:"dynaTraceMonitor",
initializedModules:"",dataDtConfig:"string"==typeof ud?ud:""};Sb().cfg=ic;let gd=ic.dataDtConfig;gd&&!gd.includes("#CONFIGSTRING")&&(B(gd,ic),Ib("domain"),Ib("auto"),Ib("app"),Kb(ic));let wd=(ic.dataDtConfig||"").includes("#CONFIGSTRING")?null:ic;if(wd=v(document.currentScript,wd),wd)for(let rd in wd)if(Object.prototype.hasOwnProperty.call(wd,rd)){V=rd;Za=ic;va=V;var Lb=wd[V];"undefined"!=typeof Lb&&(Za[va]=Lb)}let Tc=A();try{let rd=(Lb=uf)&&Lb.getItem(Tc);if(rd){let pe=zd(rd),Xd=B(pe.config||""),
Of=ic.lastModification||"0",Oi=Number.parseInt((Xd.lastModification||pe.lastModification||"0").substring(0,13)),bi="string"==typeof Of?Number.parseInt(Of.substring(0,13)):Of;if(!(Of&&Oi<bi)&&(Xd.agentUri=Xd.agentUri||La("agentUri"),Xd.featureHash=pe.featureHash||La("featureHash"),Xd.agentUri&&Kb(Xd),m(Xd,!0),$b(Xd),Tb(Xd),Oi>(+ic.lastModification||0))){let Pi=fa("auto"),ci=fa("legacy");ic=Bb(Xd);ic.auto=Pi?"1":"0";ic.legacy=ci?"1":"0"}}}catch(rd){}m(ic);try{let rd=ic.ign;if(rd&&(new RegExp(rd)).test(Ma.location.href)){var Pb=
(delete Ma.dT_,!1);break a}}catch(rd){}if(vc()){{var fd=navigator.userAgent;let pe=fd.lastIndexOf(Bg);if(-1===pe)var Kd={};else{var ff=fd.substring(pe).matchAll(Sd);if(ff){fd={};for(var wf of ff){let [,Xd,Of]=wf;fd[Xd]=Of}Kd=fd}else Kd={}}}wf=Kd;let rd;for(rd in wf)if(Object.prototype.hasOwnProperty.call(wf,rd)&&wf[rd]){Kd=ic;ff=rd;var Be=wf[rd];"undefined"!=typeof Be&&(Kd[ff]=Be)}Bb(ic)}var gf=La("cuc"),Qe,Yd;let Dd=null===(Yd=null===(Qe=Ma.dT_)||void 0===Qe?void 0:Qe.gCP)||void 0===Yd?void 0:Yd.call(Qe);
if(gf&&Dd&&gf!==Dd)throw Error("platformPrefix and cuc do not match!");Pb=!0}if(!Pb)throw Error("Error during config initialization");D();sh=Sb().disabled;var Jc;if(!(Jc=La("agentLocation")))a:{let ud=La("agentUri");if(ud||document.currentScript){let wd=ud||document.currentScript.src;if(wd){let Tc=wd.includes("_complete")?2:1,Dd=wd.lastIndexOf("/");for(Pb=0;Pb<Tc&&-1!==Dd;Pb++)wd=wd.substring(0,Dd),Dd=wd.lastIndexOf("/");Jc=wd;break a}}let gd=location.pathname;Jc=gd.substring(0,gd.lastIndexOf("/"))}Nf=
Jc;if(ee=nc(),"true"===U("dtUseDebugAgent")&&!ee.includes("dbg")&&(ee=La("debugName")||ee+"dbg"),!fa("auto")&&!fa("legacy")&&!sh){let ud=La("agentUri")||rc(La("featureHash"));fa("async")||"complete"===document.readyState?oc(ud,fa("async"),void 0,void 0,"dtjsagent"):(document.write(`<script id="${"dtjsagentdw"}" type="text/javascript" src="${ud}" crossorigin="anonymous"><\/script>`),document.getElementById("dtjsagentdw")||oc(ud,fa("async"),void 0,void 0,"dtjsagent"))}U(Ed())&&fb("cooO",!0);Ra();fb("pVO",
!!sb("dt-pVO"));K(Ja);Rd=18E5;Ag=Ea("hvt")||216E5;let xf=hb();ne=void 0;xf&&"ea"in xf&&(ne=xf.gITAID());mb(zc,[ne||1])}catch(Wd){"platformPrefix and cuc do not match!"===Wd.message?(Jc=Ma.dT_,Ma.dT_={minAgentVersion:Jc.minAgentVersion,platformPrefix:(null===(y=Jc.gCP)||void 0===y?void 0:y.call(Jc))||Jc.platformPrefix,platformCookieOffset:Jc.platformCookieOffset},vf("[CookiePrefix/initCode] cuc and platformPrefix do not match, disabling agent!")):(delete Ma.dT_,vf("JsAgent initCode initialization failed!",
Wd))}})()})();
</script><script type="text/javascript" src="/ruxitagentjs_A7NVfgqrux_10333260303165926.js" defer="defer"></script><script async src="https://www.googletagmanager.com/gtag/js?id=G-GS1KB7R8DY"></script>
<script>window.dataLayer = window.dataLayer || [];function gtag(){dataLayer.push(arguments)};gtag("js", new Date());gtag("set", "developer_id.dMDhkMT", true);gtag("config", "G-GS1KB7R8DY", {"groups":"default","page_placeholder":"PLACEHOLDER_page_location","allow_ad_personalization_signals":false});gtag("config", "UA-17192165-1", {"groups":"default","anonymize_ip":true,"page_placeholder":"PLACEHOLDER_page_path","allow_ad_personalization_signals":false});gtag("config", "G-MFMY1F6D3V", {"groups":"default","page_placeholder":"PLACEHOLDER_page_location","allow_ad_personalization_signals":false});</script>
<meta name="description" content="The style guide features formatting, spelling, punctuation, capitalization, grammar, and language guidelines." />
<link rel="canonical" href="https://www.energy.gov/cmei/communicationstandards/style-guide-full-text" />
<link rel="shortlink" href="https://www.energy.gov/node/1246196" />
<meta name="generator" content="Drupal 8 (http://drupal.org)" />
<link rel="image_src" href="https://www.energy.gov/sites/default/files/2025-04/white-fallback_0.png" />
<meta name="referrer" content="strict-origin-when-cross-origin" />
<meta property="og:site_name" content="Energy.gov" />
<meta property="og:type" content="article" />
<meta property="og:url" content="https://www.energy.gov/cmei/communicationstandards/style-guide-full-text" />
<meta property="og:title" content="Style Guide Full Text" />
<meta property="og:description" content="The style guide features formatting, spelling, punctuation, capitalization, grammar, and language guidelines." />
<meta property="og:image" content="https://www.energy.gov/sites/default/files/2025-04/white-fallback_0.png" />
<meta property="og:image:url" content="https://www.energy.gov/sites/default/files/2025-04/white-fallback_0.png" />
<meta property="og:image:type" content="image/png" />
<meta property="og:image:alt" content="Energy.gov Search Default Thumbnail" />
<meta property="og:street_address" content="1000 Independence Ave. SW" />
<meta property="og:locality" content="Washington D.C." />
<meta property="og:postal_code" content="20585" />
<meta property="og:country_name" content="USA" />
<meta property="og:phone_number" content="202-586-5000" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:description" content="The style guide features formatting, spelling, punctuation, capitalization, grammar, and language guidelines." />
<meta name="twitter:site" content="@ENERGY" />
<meta name="twitter:title" content="Style Guide Full Text" />
<meta name="twitter:image" content="https://www.energy.gov/sites/default/files/2025-04/white-fallback_0.png" />
<meta name="twitter:image:alt" content="Energy.gov Search Default Thumbnail" />
<meta name="Generator" content="Drupal 10 (https://www.drupal.org)" />
<meta name="MobileOptimized" content="width" />
<meta name="HandheldFriendly" content="true" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<style>.page-hero, .topic-banner-wrapper .hero { background-image: linear-gradient(180deg, transparent 33%, rgba(41, 41, 41, .7)), url(https://www.energy.gov/) !important } .page-title { color: #ffffff !important }</style>
<link rel="icon" href="/themes/custom/energy_gov/favicon.ico" type="image/vnd.microsoft.icon" />
<link rel="alternate" hreflang="en" href="https://www.energy.gov/cmei/communicationstandards/style-guide-full-text" />
<script src="/sites/default/files/google_tag/eere.energy.gov/google_tag.script.js?tetb3h" defer></script>
<script src="/sites/default/files/google_tag/energygov/google_tag.script.js?tetb3h" defer></script>

      <title>Style Guide Full Text | Department of Energy</title>
      <link rel="stylesheet" media="all" href="/sites/default/files/css/css_LkTfg4gFFJiDYtjShP3-8OurdtoCt--dT9VRhTOyBNU.css?delta=0&amp;language=en&amp;theme=energy_gov&amp;include=eJxNi20KgDAMQy9U7JGkujCKdZO1ft1eFAX_hLw8MtYGTm1dxLqkYjUTClo--_E2v_7tuW4s7ggnHGFapu__IrkGdk3oxdCCtWiQnx6YeRAHbYrd-clurmk1XNoANJo" />
<link rel="stylesheet" media="all" href="/sites/default/files/css/css_bdvq-2FAuhqI6RxdnSZfNCLCrx2YOWNEvwtdCNC3ccs.css?delta=1&amp;language=en&amp;theme=energy_gov&amp;include=eJxNi20KgDAMQy9U7JGkujCKdZO1ft1eFAX_hLw8MtYGTm1dxLqkYjUTClo--_E2v_7tuW4s7ggnHGFapu__IrkGdk3oxdCCtWiQnx6YeRAHbYrd-clurmk1XNoANJo" />

        
          <!-- Font Awesome includes classic solid, regular, brands, Duotone solid & regular. DOE custom icons also included below. -->
          <script defer src="/themes/custom/energy_gov/js/fontawesome/brands.js"></script>
          <script defer src="/themes/custom/energy_gov/js/fontawesome/solid.js"></script>
          <script defer src="/themes/custom/energy_gov/js/fontawesome/regular.js"></script>
          <script defer src="/themes/custom/energy_gov/js/fontawesome/duotone.js"></script>
          <script defer src="/themes/custom/energy_gov/js/fontawesome/duotone-regular.js"></script>
          <script defer src="/themes/custom/energy_gov/js/fontawesome/custom-icons.js"></script>
          <script defer src="/themes/custom/energy_gov/js/fontawesome/fontawesome.js"></script>
  </head>
  <body>
        <div id="skip-link-container">
      <a href="#main-content" class="visually-hidden focusable skip-link">
        Skip to main content
      </a>
    </div>
    <noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-WZQX9F4" height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript><noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-PCSS9JS&gtm_auth=rQjC8QLyy7zCI51ldZjEbA&gtm_preview=env-3&gtm_cookies_win=x" height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
      <div class="dialog-off-canvas-main-canvas" data-off-canvas-main-canvas>
    

      <section id="usa-govt-website-banner" class="usa-banner" aria-label="Official website of the United States government">
      
<div class="usa-banner">
  <div class="usa-accordion">
    <header class="usa-banner__header">
      <div class="usa-banner__inner">
        <div class="grid-col-auto">
          <img class="usa-banner__header-flag" aria-hidden="true" src="/themes/custom/energy_gov/assets/img/us_flag_small.png" alt="">
        </div>
        <div class="grid-col-fill tablet:grid-col-auto" aria-hidden="true">
          <p class="usa-banner__header-text">An official website of the United States government</p>
          <p class="usa-banner__header-action" aria-hidden="true">Here's how you know</p>
        </div>
        <button type="button" class="usa-accordion__button usa-banner__button" aria-expanded="false" aria-controls="gov-banner">
          <span class="usa-banner__button-text">Here's how you know</span>
        </button>
      </div>
    </header>
    <div class="usa-banner__content usa-accordion__content" id="gov-banner" hidden="">
      <div class="grid-row grid-gap-lg">
        <div class="usa-banner__guidance tablet:grid-col-6">
          <img class="usa-banner__icon usa-media-block__img" src="/themes/custom/energy_gov/assets/img/icon-dot-gov.svg" alt="Dot gov">
          <div class="usa-media-block__body">
            <p>
              <strong>Official websites use .gov</strong>
              <br>
              A <strong>.gov</strong> website belongs to an official government organization in the United States.
            </p>
          </div>
        </div>
        <div class="usa-banner__guidance tablet:grid-col-6">
          <img class="usa-banner__icon usa-media-block__img" src="/themes/custom/energy_gov/assets/img/icon-https.svg" alt="Https">
          <div class="usa-media-block__body">
            <p>
              <strong>Secure .gov websites use HTTPS</strong>
              <br>
              A <strong>lock</strong> (   <span class="icon-lock">
    <svg xmlns="http://www.w3.org/2000/svg" width="52" height="64" viewBox="0 0 52 64" class="usa-banner__lock-image" role="img" aria-labelledby="banner-lock-description" focusable="false">
      <title id="banner-lock-title">Lock</title>
      <desc id="banner-lock-description">Locked padlock</desc>
      <path fill="#000000" fill-rule="evenodd" d="M26 0c10.493 0 19 8.507 19 19v9h3a4 4 0 0 1 4 4v28a4 4 0 0 1-4 4H4a4 4 0 0 1-4-4V32a4 4 0 0 1 4-4h3v-9C7 8.507 15.507 0 26 0zm0 8c-5.979 0-10.843 4.77-10.996 10.712L15 19v9h22v-9c0-6.075-4.925-11-11-11z"/>
    </svg>
  </span>
 ) or <strong>https://</strong> means you’ve safely connected to the .gov website. Share sensitive information only on official, secure websites.
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

    </section>
  
<div class="usa-overlay"></div>

  <header class="usa-header usa-header--extended usa-header--extended-megamenu" id="header" role="banner">

    <div id="header-container">
    
    
      <div class="region region-header usa-navbar">
      
<button id="megamenu-hamburger-btn" class="usa-menu-btn megamenu-mobile-btn" type="button" aria-label="Main Menu"></button>

  <div class="usa-logo site-logo" id="logo">

    <em class="usa-logo__text">
          <a class="logo-img" href="/" accesskey="1" title="Home" aria-label="Home">
        <img src="/themes/custom/energy_gov/img/logo-image.png" alt="Home" />
      </a>
    
    
        </em>

  </div>

  <div id="search-btn-container-mobile" class="usa-accordion search-mobile">
    <button type="button" id="megamenu-search-btn-mobile" class="search usa-accordion__button usa-current" aria-expanded="false" aria-controls="search-nav" aria-label="search"></button>

    <div id="search-nav-mobile" class="usa-nav__submenu search-nav-mobile-class" hidden>
      <section aria-label="Search form">
        <div class="energy-global-search-mobile"></div>
      </section>
    </div>
  </div>


</div>


      <nav class="usa-nav" role="navigation">
                <div class="usa-nav__inner">
          
          
          
          
                          
  
        
        
  
            <ul id="main-megamenu" class="usa-nav__primary usa-accordion">
    
    
    
    
                                  
                            
        
          <li class="usa-nav__primary-item">       

                                     <button type="button" class="usa-accordion__button primary-btn " aria-expanded="false" aria-controls="extended-mega-nav-section-1-1-1">
                <span>Policy &amp; Priorities</span>
              </button>
          
                        
        
        
  
              
                         
<ul id="extended-mega-nav-section-1-1-1" class="usa-nav__submenu usa-megamenu usa-accordion" hidden="" aria-label="Policy &amp; Priorities">
    <li class="mobile-back-btn-secondary-wrapper">
        <button type="button" class="mobile-back-btn-secondary">
        <span>Main Menu</span>
        </button>
    </li>
        <li role="none" class="extended-mega-nav-section__header">
        <span class="extended-mega-nav-section__title">Policy &amp; Priorities</span>
                                  
            <div class="usa-nav__submenu-item extended-mega-nav-section__view-all">
                <a href="/topics/priorities" data-drupal-link-system-path="node/4851536">View all Policy &amp; Priorities</a>
                <svg aria-hidden="true" focusable="false" width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M14.965 8.565C15.2775 8.2525 15.2775 7.745 14.965 7.4325L11.765 4.2325C11.4525 3.92 10.945 3.92 10.6325 4.2325C10.32 4.545 10.32 5.0525 10.6325 5.365L12.4675 7.2H1.6C1.1575 7.2 0.800003 7.5575 0.800003 8C0.800003 8.4425 1.1575 8.8 1.6 8.8H12.4675L10.6325 10.635C10.32 10.9475 10.32 11.455 10.6325 11.7675C10.945 12.08 11.4525 12.08 11.765 11.7675L14.965 8.5675V8.565Z"/>
                </svg>
            </div>
                                                                                                                                                                                                            </li>
                         
                            
                            
        
        <li class="megamenu-submenu-policy--priorities-item">

            <div class="usa-nav__submenu-item">
                                <a href="/topics/increase-american-energy-production" title="Expanding domestic energy resources to support national needs and economic growth." class="link" data-drupal-link-system-path="node/4851240">                                    <span>Increase American Energy Production</span>
                    <div class="link-description"><span>Expanding domestic energy resources to support national needs and economic growth.</span></div>
                                </a>
            </div>

        </li>

                     
                            
                            
        
        <li class="megamenu-submenu-policy--priorities-item">

            <div class="usa-nav__submenu-item">
                                <a href="/topics/energy-innovation" title="Driving research and development to shape the future of energy systems and technologies." class="link" data-drupal-link-system-path="node/4851243">                                    <span>Energy Innovation</span>
                    <div class="link-description"><span>Driving research and development to shape the future of energy systems and technologies.</span></div>
                                </a>
            </div>

        </li>

                     
                            
                            
        
        <li class="megamenu-submenu-policy--priorities-item">

            <div class="usa-nav__submenu-item">
                                <a href="/topics/energy-exports" title="Supporting global energy markets through U.S. energy production and trade." class="link" data-drupal-link-system-path="node/4851242">                                    <span>Energy Exports</span>
                    <div class="link-description"><span>Supporting global energy markets through U.S. energy production and trade.</span></div>
                                </a>
            </div>

        </li>

                     
                            
                            
        
        <li class="megamenu-submenu-policy--priorities-item">

            <div class="usa-nav__submenu-item">
                                <a href="/topics/affordable-energy-consumers" title="Promoting access to reliable and cost-effective energy for households and businesses." class="link" data-drupal-link-system-path="node/4851241">                                    <span>Affordable Energy for Consumers</span>
                    <div class="link-description"><span>Promoting access to reliable and cost-effective energy for households and businesses.</span></div>
                                </a>
            </div>

        </li>

                     
                            
                            
        
        <li class="megamenu-submenu-policy--priorities-item">

            <div class="usa-nav__submenu-item">
                                <a href="/topics/strategic-petroleum-reserve-spr" title="Managing emergency fuel reserves to support national energy security." class="link" data-drupal-link-system-path="node/4851315">                                    <span>Strategic Petroleum Reserve (SPR)</span>
                    <div class="link-description"><span>Managing emergency fuel reserves to support national energy security.</span></div>
                                </a>
            </div>

        </li>

                     
                            
                            
        
        <li class="megamenu-submenu-policy--priorities-item">

            <div class="usa-nav__submenu-item">
                                <a href="/topics/nuclear-security" title="Maintaining the safety, security, and effectiveness of the nation’s nuclear deterrent." class="link" data-drupal-link-system-path="node/4851316">                                    <span>Nuclear Security</span>
                    <div class="link-description"><span>Maintaining the safety, security, and effectiveness of the nation’s nuclear deterrent.</span></div>
                                </a>
            </div>

        </li>

                     
                            
                            
        
        <li class="megamenu-submenu-policy--priorities-item">

            <div class="usa-nav__submenu-item">
                                <a href="/topics/nuclear-energy" title="Advancing nuclear technologies to support energy, security, and innovation goals." class="link" data-drupal-link-system-path="node/4851317">                                    <span>Nuclear Energy</span>
                    <div class="link-description"><span>Advancing nuclear technologies to support energy, security, and innovation goals.</span></div>
                                </a>
            </div>

        </li>

                     
                            
                            
        
        <li class="megamenu-submenu-policy--priorities-item">

            <div class="usa-nav__submenu-item">
                                <a href="/topics/energy-reliability" title="Ensuring consistent and dependable energy delivery across the nation." class="link" data-drupal-link-system-path="node/4851318">                                    <span>Energy Reliability</span>
                    <div class="link-description"><span>Ensuring consistent and dependable energy delivery across the nation.</span></div>
                                </a>
            </div>

        </li>

                     
                            
                            
        
        <li class="megamenu-submenu-policy--priorities-item">

            <div class="usa-nav__submenu-item">
                                <a href="/topics/energy-infrastructure" title="Strengthening and modernizing the systems that produce, deliver, and store energy." class="link" data-drupal-link-system-path="node/4851247">                                    <span>Energy Infrastructure</span>
                    <div class="link-description"><span>Strengthening and modernizing the systems that produce, deliver, and store energy.</span></div>
                                </a>
            </div>

        </li>

            </ul>    
    
     
   
          
        
          </li>
                                  
                            
        
          <li class="usa-nav__primary-item">       

                                     <button type="button" class="usa-accordion__button primary-btn " aria-expanded="false" aria-controls="extended-mega-nav-section-2-1-2">
                <span>Leadership &amp; Organization</span>
              </button>
          
                        
        
        
  
              
                         <ul id="extended-mega-nav-section-2-1-2" class="megamenu-container--4col megamenu-container--leadership-organization usa-nav__submenu usa-megamenu usa-accordion" hidden="" aria-label="Leadership &amp; Organization">
    <li class="mobile-back-btn-secondary-wrapper">
        <button type="button" class="mobile-back-btn-secondary">
        <span>Main Menu</span>
        </button>
    </li>
    <li role="none" class="extended-mega-nav-section__header">
        <span class="extended-mega-nav-section__title">Leadership &amp; Organization</span>
                                  
            <div class="usa-nav__submenu-item extended-mega-nav-section__view-all">
                <a href="/leadership-organization" data-drupal-link-system-path="node/4857025">View all Leadership &amp; Organization</a>
                <svg aria-hidden="true" focusable="false" width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M14.965 8.565C15.2775 8.2525 15.2775 7.745 14.965 7.4325L11.765 4.2325C11.4525 3.92 10.945 3.92 10.6325 4.2325C10.32 4.545 10.32 5.0525 10.6325 5.365L12.4675 7.2H1.6C1.1575 7.2 0.800003 7.5575 0.800003 8C0.800003 8.4425 1.1575 8.8 1.6 8.8H12.4675L10.6325 10.635C10.32 10.9475 10.32 11.455 10.6325 11.7675C10.945 12.08 11.4525 12.08 11.765 11.7675L14.965 8.5675V8.565Z"/>
                </svg>
            </div>
                                                                                                                                                </li>
    <li class="megamenu--leadership-organization__leadership-block">
                    <div class="megamenu-block__leadership">
                            <div class="megamenu-block__leader">
                <a href="/person/chris-wright" tabindex="-1">
                    <img src="/sites/default/files/2025-02/CHRIS_WRIGHT_PORTRAIT_SQ.jpg"
                         alt="Chris Wright"
                         class="megamenu-block__headshot"  
                    />
                </a>
                <div class="usa-nav__submenu-item megamenu-block__leader--details">
                    <a href="/person/chris-wright">
                        Chris Wright
                        <div class="link-description">
                            <span class="megamenu-block__role">Secretary of Energy</span>
                        </div>
                    </a>
                </div>
                </div>
                                        <div class="megamenu-block__leader">
                    <a href="/person/james-danly" tabindex="-1">
                        <img src="/sites/default/files/2025-07/JAMES_DANLY_SQ.jpg"
                            alt="James Danly Deputy Secretary of Energy Headshot"
                            class="megamenu-block__headshot"  
                        />
                    </a>
                <div class="usa-nav__submenu-item megamenu-block__leader--details">
                    <a href="/person/james-danly">
                        James Danly
                        <div class="link-description">
                            <span class="megamenu-block__role">Deputy Secretary of Energy</span>
                        </div>
                    </a>
                </div>
                </div>
                        </div>
            </li>
    <li class="megamenu--leadership-organization__links">
        <span class="visually-hidden">Leadership & Organization Pages</span>
        <ul>
                                                         
                                                                
                                                                
                                        <li class="megamenu-submenu-leadership--organization-item">
                        <div class="usa-nav__submenu-item">
                                                        <a href="/staff-offices" title="Supporting DOE operations through policy, administration, and mission services." class="link" data-drupal-link-system-path="node/4857044">                                                            <span>Staff Offices</span>
                                <br><div class="link-description"><span>Supporting DOE operations through policy, administration, and mission services.</span></div>
                                                        </a>
                        </div>
                    </li>
                                             
                                                                
                                                                
                                        <li class="megamenu-submenu-leadership--organization-item">
                        <div class="usa-nav__submenu-item">
                                                        <a href="/science-and-innovation-program-offices" title="Leading scientific research and technology development across priority areas." class="link" data-drupal-link-system-path="node/4857056">                                                            <span>Science and Innovation Program Offices</span>
                                <br><div class="link-description"><span>Leading scientific research and technology development across priority areas.</span></div>
                                                        </a>
                        </div>
                    </li>
                                             
                                                                
                                                                
                                        <li class="megamenu-submenu-leadership--organization-item">
                        <div class="usa-nav__submenu-item">
                                                        <a href="/independent-offices" title="Providing oversight and specialized functions across key areas of the Department." class="link" data-drupal-link-system-path="node/4857043">                                                            <span>Independent Offices</span>
                                <br><div class="link-description"><span>Providing oversight and specialized functions across key areas of the Department.</span></div>
                                                        </a>
                        </div>
                    </li>
                                             
                                                                
                                                                
                                        <li class="megamenu-submenu-leadership--organization-item">
                        <div class="usa-nav__submenu-item">
                                                        <a href="/energy-program-offices" title="Managing programs that support energy production, efficiency, and delivery." class="link" data-drupal-link-system-path="node/4857055">                                                            <span>Energy Program Offices</span>
                                <br><div class="link-description"><span>Managing programs that support energy production, efficiency, and delivery.</span></div>
                                                        </a>
                        </div>
                    </li>
                                             
                                                                
                                                                
                                        <li class="megamenu-submenu-leadership--organization-item">
                        <div class="usa-nav__submenu-item">
                                                        <a href="/national-laboratories" title="Advancing research and innovation through DOE’s National Laboratory System." class="link" data-drupal-link-system-path="node/4857057">                                                            <span>National Laboratories</span>
                                <br><div class="link-description"><span>Advancing research and innovation through DOE’s National Laboratory System.</span></div>
                                                        </a>
                        </div>
                    </li>
                                             
                                                                
                                                                
                                        <li class="megamenu-submenu-leadership--organization-item">
                        <div class="usa-nav__submenu-item">
                                                        <a href="/nnsa/national-nuclear-security-administration" title="Overseeing national security efforts related to nuclear safety and defense." class="link" data-drupal-link-system-path="node/4846918">                                                            <span>Nuclear Security Offices</span>
                                <br><div class="link-description"><span>Overseeing national security efforts related to nuclear safety and defense.</span></div>
                                                        </a>
                        </div>
                    </li>
                                    </ul>
    </li>
    <li class="megamenu--leadership-organization__featured-office">
                    <div class="usa-nav__submenu-item megamenu-block__spotlight">
                <header><i class="fa-solid fa-atom"></i> Featured</header>
                <a href="/hgeo/hydrocarbons-and-geothermal-energy-office">
                                        <div class="megamenu-block__spotlight-image-wrapper">
                        <img src="/sites/default/files/2025-12/Oil%20and%20gas%20transport.jpeg" alt="Hydrocarbons and Geothermal Energy Office" />
                    </div>
                                        <span>Hydrocarbons and Geothermal Energy Office</span>
                </a>
            </div>
            </li>
</ul>    
    
     
   
          
        
          </li>
                                  
                            
        
          <li class="usa-nav__primary-item">       

                                     <button type="button" class="usa-accordion__button primary-btn " aria-expanded="false" aria-controls="extended-mega-nav-section-3-1-3">
                <span>Topics</span>
              </button>
          
                        
        
        
  
              
                         <ul id="extended-mega-nav-section-3-1-3" class="usa-nav__submenu usa-megamenu usa-accordion" hidden="" aria-label="Topics">
    <li class="mobile-back-btn-secondary-wrapper">
        <button type="button" class="mobile-back-btn-secondary">
        <span>Main Menu</span>
        </button>
    </li>
    <li role="none" class="extended-mega-nav-section__header">
        <span class="extended-mega-nav-section__title">Topics</span>
                                   
            <div class="usa-nav__submenu-item extended-mega-nav-section__view-all">
                <a href="https://www.energy.gov/topics">View all Topics</a>
                <svg aria-hidden="true" focusable="false" width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M14.965 8.565C15.2775 8.2525 15.2775 7.745 14.965 7.4325L11.765 4.2325C11.4525 3.92 10.945 3.92 10.6325 4.2325C10.32 4.545 10.32 5.0525 10.6325 5.365L12.4675 7.2H1.6C1.1575 7.2 0.800003 7.5575 0.800003 8C0.800003 8.4425 1.1575 8.8 1.6 8.8H12.4675L10.6325 10.635C10.32 10.9475 10.32 11.455 10.6325 11.7675C10.945 12.08 11.4525 12.08 11.765 11.7675L14.965 8.5675V8.565Z"/>
                </svg>
            </div>
                                                                                                                                                </li>
                         
                            
                            
        
        <li class="megamenu-submenu-topics-item">

                                                <div class="usa-nav__submenu-item">
                <a href="/topics/energy-sources">
                    <div  data-region="primary_menu" class="megamenu-topic" style="background-image: url(/sites/default/files/2026-04/energy-sources.png)">
                                            

            <div class="field field--fontawesome_icon_formatter field--field_font_awesome_icon field field--name-field-font-awesome-icon field--type-fontawesome-icon field--label-hidden field__item"><div class="fontawesome-icons">
      <div class="fontawesome-icon">
  <i class="fa-solid fa-oil-well" data-fa-transform="" data-fa-mask="" style="--fa-primary-color: #000000; --fa-secondary-color: #000000;"></i>
</div>

  </div>
</div>
      
                                        <span>Energy Sources</span>
                                            <div class="link-description"><span>Learn about the resources and technologies used to produce power across the United States.</span></div>
                                        </div>
                </a>
            </div>
        </li>

                     
                            
                            
        
        <li class="megamenu-submenu-topics-item">

                                                <div class="usa-nav__submenu-item">
                <a href="/topics/energy-usage">
                    <div  data-region="primary_menu" class="megamenu-topic" style="background-image: url(/sites/default/files/2026-04/energy-usage.png)">
                                            

            <div class="field field--fontawesome_icon_formatter field--field_font_awesome_icon field field--name-field-font-awesome-icon field--type-fontawesome-icon field--label-hidden field__item"><div class="fontawesome-icons">
      <div class="fontawesome-icon">
  <i class="fa-solid fa-bolt-lightning" data-fa-transform="" data-fa-mask="" style="--fa-primary-color: #000000; --fa-secondary-color: #000000;"></i>
</div>

  </div>
</div>
      
                                        <span>Energy Usage</span>
                                            <div class="link-description"><span>Explore how energy is consumed in homes, businesses, transportation, and industry.</span></div>
                                        </div>
                </a>
            </div>
        </li>

                     
                            
                            
        
        <li class="megamenu-submenu-topics-item">

                                                <div class="usa-nav__submenu-item">
                <a href="/topics/policy">
                    <div  data-region="primary_menu" class="megamenu-topic" style="background-image: url(/sites/default/files/2026-04/energy-policy.png)">
                                            

            <div class="field field--fontawesome_icon_formatter field--field_font_awesome_icon field field--name-field-font-awesome-icon field--type-fontawesome-icon field--label-hidden field__item"><div class="fontawesome-icons">
      <div class="fontawesome-icon">
  <i class="fa-solid fa-building-columns" data-fa-transform="" data-fa-mask="" style="--fa-primary-color: #000000; --fa-secondary-color: #000000;"></i>
</div>

  </div>
</div>
      
                                        <span>Policy</span>
                                            <div class="link-description"><span>Understand the laws, regulations, and strategies that guide energy decisions and national priorities.</span></div>
                                        </div>
                </a>
            </div>
        </li>

                     
                            
                            
        
        <li class="megamenu-submenu-topics-item">

                                                <div class="usa-nav__submenu-item">
                <a href="/topics/science-innovation">
                    <div  data-region="primary_menu" class="megamenu-topic" style="background-image: url(/sites/default/files/2026-04/energy-innovation.png)">
                                            

            <div class="field field--fontawesome_icon_formatter field--field_font_awesome_icon field field--name-field-font-awesome-icon field--type-fontawesome-icon field--label-hidden field__item"><div class="fontawesome-icons">
      <div class="fontawesome-icon">
  <i class="fa-solid fa-atom" data-fa-transform="" data-fa-mask="" style="--fa-primary-color: #000000; --fa-secondary-color: #000000;"></i>
</div>

  </div>
</div>
      
                                        <span>Science &amp; Innovation</span>
                                            <div class="link-description"><span>See how research, discovery, and new technologies are shaping the future of energy.</span></div>
                                        </div>
                </a>
            </div>
        </li>

                     
                            
                            
        
        <li class="megamenu-submenu-topics-item">

                                                <div class="usa-nav__submenu-item">
                <a href="/topics/economy">
                    <div  data-region="primary_menu" class="megamenu-topic" style="background-image: url(/sites/default/files/2026-04/energy-economy.png)">
                                            

            <div class="field field--fontawesome_icon_formatter field--field_font_awesome_icon field field--name-field-font-awesome-icon field--type-fontawesome-icon field--label-hidden field__item"><div class="fontawesome-icons">
      <div class="fontawesome-icon">
  <i class="fa-solid fa-chart-line" data-fa-transform="" data-fa-mask="" style="--fa-primary-color: #000000; --fa-secondary-color: #000000;"></i>
</div>

  </div>
</div>
      
                                        <span>Economy</span>
                                            <div class="link-description"><span>Discover how energy affects jobs, markets, investment, and economic growth.</span></div>
                                        </div>
                </a>
            </div>
        </li>

                     
                            
                            
        
        <li class="megamenu-submenu-topics-item">

                                                <div class="usa-nav__submenu-item">
                <a href="/topics/national-security-safety">
                    <div  data-region="primary_menu" class="megamenu-topic" style="background-image: url(/sites/default/files/2026-04/energy-security.png)">
                                            

            <div class="field field--fontawesome_icon_formatter field--field_font_awesome_icon field field--name-field-font-awesome-icon field--type-fontawesome-icon field--label-hidden field__item"><div class="fontawesome-icons">
      <div class="fontawesome-icon">
  <i class="fa-solid fa-shield-halved" data-fa-transform="" data-fa-mask="" style="--fa-primary-color: #000000; --fa-secondary-color: #000000;"></i>
</div>

  </div>
</div>
      
                                        <span>National Security &amp; Safety</span>
                                            <div class="link-description"><span>Learn how energy systems support resilience, infrastructure protection, and public safety.</span></div>
                                        </div>
                </a>
            </div>
        </li>

            </ul>    
    
     
   
          
        
          </li>
                                  
                            
        
          <li class="usa-nav__primary-item">       

                                     <button type="button" class="usa-accordion__button primary-btn " aria-expanded="false" aria-controls="extended-mega-nav-section-4-1-4">
                <span>News &amp; Events</span>
              </button>
          
                        
        
        
  
              
                         <ul id="extended-mega-nav-section-4-1-4" class="usa-nav__submenu usa-megamenu usa-accordion" hidden="" aria-label="News &amp; Events">
    <li class="mobile-back-btn-secondary-wrapper">
        <button type="button" class="mobile-back-btn-secondary">
        <span>Main Menu</span>
        </button>
    </li>
    <li role="none" class="extended-mega-nav-section__header">
        <span class="extended-mega-nav-section__title">News &amp; Events</span>
                                  
            <div class="usa-nav__submenu-item extended-mega-nav-section__view-all">
                <a href="/newsroom" data-drupal-link-system-path="node/4847098">View all News &amp; Events</a>
                <svg aria-hidden="true" focusable="false" width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M14.965 8.565C15.2775 8.2525 15.2775 7.745 14.965 7.4325L11.765 4.2325C11.4525 3.92 10.945 3.92 10.6325 4.2325C10.32 4.545 10.32 5.0525 10.6325 5.365L12.4675 7.2H1.6C1.1575 7.2 0.800003 7.5575 0.800003 8C0.800003 8.4425 1.1575 8.8 1.6 8.8H12.4675L10.6325 10.635C10.32 10.9475 10.32 11.455 10.6325 11.7675C10.945 12.08 11.4525 12.08 11.765 11.7675L14.965 8.5675V8.565Z"/>
                </svg>
            </div>
                        </li>
    <li class="megamenu--news-events__press-releases">
        <div class="views-element-container"><div class="megamenu--news-events--press-releases megamenu--news-events__press-releases js-view-dom-id-1cfbb6e8284caba7b00df1012c6ff74aa95bfd96b2dbcb08732cae46f210a8fe">
  
  
  

      <header>
      Press Releases
    </header>
  
  
  

      <div class="views-row"><div class="views-field views-field-field-display-date"><div class="field-content"><time datetime="2026-05-08T10:00:01Z">May 8, 2026</time>
</div></div><div class="views-field views-field-title usa-nav__submenu-item"><span class="field-content"><a href="/nnsa/articles/nnsa-removes-highly-enriched-uranium-venezuela-reducing-risk-south-america-and-us" hreflang="en">NNSA Removes Highly Enriched Uranium from Venezuela, Reducing Risk to South America and the U.S. Homeland</a></span></div><div class="views-field views-field-field-summary"><div class="field-content">The safe removal of all enriched uranium from Venezuela sends another signal to the… <a href="/nnsa/articles/nnsa-removes-highly-enriched-uranium-venezuela-reducing-risk-south-america-and-us" class="views-more-link">Read more</a></div></div><div class="views-field views-field-field-topic-page"><div class="field-content"><a href="/topics/nuclear-security-legacy" hreflang="en">Nuclear Security Legacy</a></div></div></div>

    

  
  <div class="more-link"><a href="/search?page=0&amp;sort_by=date&amp;f%5B0%5D=article_type%3A1&amp;f%5B1%5D=offices_rest%3AEnergy.gov">View all Press Releases</a></div>


  
  
</div>
</div>

    </li>
    <li class="megamenu--news-events__fact-sheets">
        <div class="views-element-container"><div class="megamenu--news-events--fact-sheets megamenu--news-events__fact-sheets js-view-dom-id-2368a6423b0bedf7720f8814c37b1904c151ec0653398dad0d9c83808d4767b9">
  
  
  

      <header>
      Fact Sheets
    </header>
  
  
  

      <div class="views-row"><div class="views-field views-field-title usa-nav__submenu-item"><span class="field-content"><a href="/articles/fact-sheet-department-energy-ensuring-affordable-energy-access-ohio-while-powering-future" hreflang="en">FACT SHEET: The Department of Energy is Ensuring Affordable Energy Access in Ohio While Powering the Future of AI</a></span></div><div class="views-field views-field-field-topic-page"><div class="field-content"></div></div></div>
    <div class="views-row"><div class="views-field views-field-title usa-nav__submenu-item"><span class="field-content"><a href="/articles/fact-sheet-energy-department-prevented-blackouts-saved-american-lives-during-winter-storms" hreflang="en">FACT SHEET: Energy Department Prevented Blackouts &amp; Saved American Lives During Winter Storms</a></span></div><div class="views-field views-field-field-topic-page"><div class="field-content"></div></div></div>
    <div class="views-row"><div class="views-field views-field-title usa-nav__submenu-item"><span class="field-content"><a href="/articles/fact-sheet-department-energy-ending-war-beautiful-clean-coal" hreflang="en">FACT SHEET: The Department of Energy Is Ending The War On Beautiful, Clean Coal</a></span></div><div class="views-field views-field-field-topic-page"><div class="field-content"></div></div></div>

    

  
  <div class="more-link"><a href="/search?page=0&amp;f%5B0%5D=bundle_union%3Aarticle&amp;f%5B1%5D=article_type%3A1381215&amp;f%5B2%5D=offices_rest%3AEnergy.gov">View all Fact Sheets</a></div>


  
  
</div>
</div>

    </li>
    <li class="megamenu--news-events__upcoming-events">
        <div class="views-element-container"><div class="megamenu--news-events--upcoming-events megamenu--news-events__upcoming-events js-view-dom-id-3cec8b8ef1a6ee4d6449fcf776c451b2d84c23899fe2ef98964be6a25389d838">
  
  
  

      <header>
      Upcoming Events
    </header>
  
  
  

      <div class="views-row"><div class="views-field views-field-field-event-date"><div class="field-content">May 13</div></div><div class="views-field views-field-field-event-date-1"><div class="field-content">6:00PM EDT - 8:00PM EDT</div></div><div class="views-field views-field-title usa-nav__submenu-item"><span class="field-content"><a href="/em/events/oak-ridge-em-ssab-meeting-may-2026" hreflang="en">Oak Ridge EM SSAB Meeting (May 2026)</a></span></div></div>
    <div class="views-row"><div class="views-field views-field-field-event-date"><div class="field-content">May 13</div></div><div class="views-field views-field-field-event-date-1"><div class="field-content">7:30PM EDT - 10:00PM EDT</div></div><div class="views-field views-field-title usa-nav__submenu-item"><span class="field-content"><a href="/etec/events/connect-series-community-update-2" hreflang="en">CONNECT Series Community Update #2</a></span></div></div>
    <div class="views-row"><div class="views-field views-field-field-event-date"><div class="field-content">May 14</div></div><div class="views-field views-field-field-event-date-1"><div class="field-content">3:00PM EDT - 4:30PM EDT</div></div><div class="views-field views-field-title usa-nav__submenu-item"><span class="field-content"><a href="/indianenergy/events/microgrids-and-distribution-infrastructure-0" hreflang="en">Microgrids and Distribution Infrastructure</a></span></div></div>

    

  
  <div class="more-link"><a href="/search-events?keywords=events&amp;page=0&amp;startDate%5Bvalue%5D=now">View all Events</a></div>


  
  
</div>
</div>

    </li>
</ul>    
    
     
   
          
        
          </li>
                                  
                            
        
          <li class="usa-nav__primary-item">       

                                     <button type="button" class="usa-accordion__button primary-btn " aria-expanded="false" aria-controls="extended-mega-nav-section-5-1-5">
                <span>About</span>
              </button>
          
                        
        
        
  
              
                         


<ul id="extended-mega-nav-section-5-1-5" class="megamenu-container--4col megamenu-container--about usa-nav__submenu usa-megamenu usa-accordion" hidden="" aria-label="About">
    <li class="mobile-back-btn-secondary-wrapper">
        <button type="button" class="mobile-back-btn-secondary">
        <span>Main Menu</span>
        </button>
    </li>
    <li role="none" class="extended-mega-nav-section__header">
        <span class="extended-mega-nav-section__title">About</span>
                                                                                            </li>
                                                             
                            
        
        <li class="megamenu-submenu-about-item">
            <div class="usa-nav__submenu-item">
                                <a href="/mission" title="Supporting USA energy, science, and security priorities." class="link" data-drupal-link-system-path="node/4857024">                
                                    

            <div class="field field--fontawesome_icon_formatter field--field_font_awesome_icon field field--name-field-font-awesome-icon field--type-fontawesome-icon field--label-hidden field__item"><div class="fontawesome-icons">
      <div class="fontawesome-icon">
  <i class="fa-solid fa-atom" data-fa-transform="" data-fa-mask="" style="--fa-primary-color: #000000; --fa-secondary-color: #000000;"></i>
</div>

  </div>
</div>
      
                
                                    <span>Mission</span>
                    <br><div class="link-description"><span>Supporting USA energy, science, and security priorities.</span></div>
                                </a>
            </div>
                    </li>

    
                                                                     
                            
        
        <li class="megamenu-submenu-about-item">
            <div class="usa-nav__submenu-item">
                                <a href="/lm/history" class="link" data-drupal-link-system-path="node/4847006">                
                                    

            <div class="field field--fontawesome_icon_formatter field--field_font_awesome_icon field field--name-field-font-awesome-icon field--type-fontawesome-icon field--label-hidden field__item"><div class="fontawesome-icons">
      <div class="fontawesome-icon">
  <i class="fa-solid fa-microscope" data-fa-transform="" data-fa-mask="" style="--fa-primary-color: #000000; --fa-secondary-color: #000000;"></i>
</div>

  </div>
</div>
      
                
                                    <span>History &amp; Achievements</span>
                                </a>
            </div>
                            <ul class="usa-nav__submenu-list">
                                                        
                            
        
        <li class="megamenu-submenu-about-item">
            <div class="usa-nav__submenu-item">
                                <a href="/doe-affiliated-nobel-prize-laureates" class="link" data-drupal-link-system-path="node/4846968">                
                                    
                
                                    <span>Nobel Prize Laureates</span>
                                </a>
            </div>
                    </li>

    
                </ul>
                    </li>

    
                                                                     
                            
        
        <li class="megamenu-submenu-about-item">
            <div class="usa-nav__submenu-item">
                                <a href="/contact-us" title="Access general information and ways to connect with the Department of Energy." class="link" data-drupal-link-system-path="node/268177">                
                                    

            <div class="field field--fontawesome_icon_formatter field--field_font_awesome_icon field field--name-field-font-awesome-icon field--type-fontawesome-icon field--label-hidden field__item"><div class="fontawesome-icons">
      <div class="fontawesome-icon">
  <i class="fa-solid fa-paper-plane" data-fa-transform="" data-fa-mask="" style="--fa-primary-color: #000000; --fa-secondary-color: #000000;"></i>
</div>

  </div>
</div>
      
                
                                    <span>Contact Us</span>
                    <br><div class="link-description"><span>Access general information and ways to connect with the Department of Energy.</span></div>
                                </a>
            </div>
                    </li>

    
                                                                     
                            
        
        <li class="megamenu-submenu-about-item">
            <div class="usa-nav__submenu-item">
                                <a href="/work-us-department-energy" class="link" data-drupal-link-system-path="node/4847097">                
                                    
                
                                    <span>Careers</span>
                                </a>
            </div>
                            <ul class="usa-nav__submenu-list">
                                                        
                            
        
        <li class="megamenu-submenu-about-item">
            <div class="usa-nav__submenu-item">
                                <a href="/internships-fellowships" class="link" data-drupal-link-system-path="node/4846914">                
                                    
                
                                    <span>Internships &amp; Fellowships</span>
                                </a>
            </div>
                    </li>

                                
                            
        
        <li class="megamenu-submenu-about-item">
            <div class="usa-nav__submenu-item">
                                <a href="/apprenticeships-workforce-development" class="link" data-drupal-link-system-path="node/4846922">                
                                    
                
                                    <span>Apprenticeships &amp; Workforce Development</span>
                                </a>
            </div>
                    </li>

                                
                            
        
        <li class="megamenu-submenu-about-item">
            <div class="usa-nav__submenu-item">
                                <a href="/careers/jobs-doe" class="link" data-drupal-link-system-path="node/4816033">                
                                    
                
                                    <span>Work at DOE</span>
                                </a>
            </div>
                    </li>

    
                </ul>
                    </li>

    
            </ul>    
    
     
   
          
        
          </li>
      
    </ul> 
                  

<div class="block block-block-content block-block-content55736c1c-e4a5-45b5-ad7d-52545b8dc012">
          
        
              

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p style="margin:0;padding:0;"><a class="usa-button usa-button--outline" href="/funding-opportunities" type="button">Funding Opportunities</a></p></div>
      
              
      </div>
         <div id="search-btn-container" class="usa-accordion search-btn-megamenu search-desktop">
          <button type="button" id="megamenu-search-btn" class="search usa-accordion__button usa-current search-icon" aria-expanded="false" aria-controls="search-nav" aria-label="Search">
          </button>

          <div id="search-nav" class="usa-nav__submenu" hidden>
            <section aria-label="Search form">
              <div class="energy-global-search"></div>
            </section>
          </div>
        </div>
       
     
   



            
            
            
                  </div>
              </nav>

        </div>
</header>


  <div class="region-header-ful-width">
      <div>
    

<div data-sitewide-alert aria-live="polite" id="block-energy-gov-sitewidealert" class="block block-sitewide-alert-block">
          
        
              <div></div>

              
      </div>

  </div>

  </div>


<main class="main-content usa-layout-docs usa-section node-type--page" id="main-content" role="main" tabindex="-1">

    <div class="grid-container">

      
      <div class="grid-row grid-gap">
        
                  <div class="usa-layout-docs__main desktop:grid-col-fill">
                                          

<div id="block-main-breadcrumbs" class="block block-system block-system-breadcrumb-block">
          
        
                  <nav class="usa-breadcrumb" role="navigation" aria-labelledby="system-breadcrumb">
        <h2 id="system-breadcrumb" class="usa-sr-only">Breadcrumb</h2>
        <ol class="usa-breadcrumb__list">
                            <li class="usa-breadcrumb__list-item">
                                            <a href="/cmei/communicationstandards" class="usa-breadcrumb__link"><span>Communication Standards and Guidelines</span></a>
                                    </li>
                            <li class="usa-breadcrumb__list-item">
                                            <a href="/cmei/communicationstandards/style-guide-full-text" class="usa-breadcrumb__link"><span>Style Guide Full Text</span></a>
                                    </li>
                    </ol>
    </nav>

              
      </div>


<div id="block-pagetitle" class="light_text_dark_background block block-core block-page-title-block">
          
        
              
  <section  class="hero hero--page">
      <div class="hero__img"></div>
      <div class="grid-container">
          <div class="hero__title">
        <h1>
<span property="schema:name" class="field field--string field--title">Style Guide Full Text</span>
</h1>
      </div>
    
        </div>
  </section>


              
      </div>
<div data-drupal-messages-fallback class="hidden"></div>
<div id="block-main-page-content" class="block block-system block-system-main-block">
          
        
        
        
      <article about="/cmei/communicationstandards/style-guide-full-text" typeof="schema:WebPage">

  
      <span property="schema:name" content="Style Guide Full Text" class="hidden"></span>


  
  <div>
    



  

<section class="layout blade-section--boxed"  id="intro-text">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      

<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p>Below is the full text of the style guide for web pages for the U.S. Department of Energy's Office of Critical Minerals and Energy Innovation (CMEI). The guide features formatting, spelling, punctuation, capitalization, grammar, and language guidelines.</p><p>Guidelines are listed alphabetically for easy reference. You may also use the&nbsp;<a href="https://www.energy.gov/eere/communicationstandards/eere-style-guide">topic index</a>&nbsp;to locate information covered in the guide.</p><p>Please reach out to the <a href="mailto:DL-EERE_Comms_Standards@ee.doe.gov">CMEI Comms Standards Team</a> with any concerns or questions.</p></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic layout-align layout-center layout-full layout-width">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#a">A</a>&nbsp;|&nbsp;<a href="#B">B</a>&nbsp;|&nbsp;<a href="#C">C</a>&nbsp;|&nbsp;<a href="#D">D</a>&nbsp;|&nbsp;<a href="#E">E</a>&nbsp;|&nbsp;<a href="#F">F</a>&nbsp;|&nbsp;<a href="#G">G</a>&nbsp;|&nbsp;<a href="#H">H</a>&nbsp;|&nbsp;<a href="#I">I</a>&nbsp;| J |&nbsp;<a href="#K">K</a>&nbsp;|&nbsp;<a href="#L">L</a>&nbsp;|&nbsp;<a href="#M">M</a>&nbsp;|&nbsp;<a href="#N">N</a>&nbsp;|&nbsp;<a href="#O">O</a>&nbsp;|&nbsp;<a href="#P">P</a>&nbsp;|&nbsp;<a href="#Q">Q</a>&nbsp;|&nbsp;<a href="#R">R</a>&nbsp;|&nbsp;<a href="#S">S</a>&nbsp;|&nbsp;<a href="#T">T</a>&nbsp;|&nbsp;<a href="#U">U</a>&nbsp;| V |&nbsp;<a href="#W">W</a>&nbsp;| X |&nbsp;<a href="#Y">Y</a>&nbsp;|&nbsp;<a href="#Z">Z</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="a">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--a">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--a">
        
        A
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="A"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--a-an">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--a-an" id="an">
        
        a, an
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="an"></a>Use "a"&nbsp;before any acronym or word that begins with a consonant sound. Use "an"&nbsp;before any acronym or word that begins with a vowel sound. An acronym is pronounced as a word (e.g., a HEPA filter); an initialism is pronounced as its letters (e.g., an NGO).</p>

<ul>
	<li>a light-water reactor; an LWR</li>
	<li>a request for proposals; an RFP</li>
	<li>a NASA astronaut</li>
	<li>a Project Management Plan; a PMP</li>
</ul>

<p>&nbsp;</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--abbreviations-acronyms-and-initialisms">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--abbreviations-acronyms-and-initialisms" id="abbreviations">
        
        abbreviations, acronyms, and initialisms
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="abbreviations"></a>An <strong>abbreviation </strong>is a shortened form of a word used in place of the full word. An <strong>acronym </strong>is a word formed from the first letters of each of the words in a phrase or name. An <strong>initialism </strong>is similar to an acronym, but it is pronounced by its letters.</p><p>Include a glossary or list of acronyms if your publication is at least 50 pages and contains at least 20 acronyms.</p><p><em>Abbreviations</em></p><ul><li>Corp.</li><li>Blvd.</li><li>FedEx</li></ul><p><em>Acronyms</em></p><ul><li>NASA</li><li>laser</li></ul><p><em>Initialisms</em></p><ul><li>DOE</li><li>CMEI</li></ul><p>Spell out the full name on first use, followed immediately by the abbreviation/acronym/initialism in parentheses. To avoid confusion, try not to use too many in a sentence or paragraph.</p><p>Avoid abbreviations, acronyms, and initialisms unless the phrase appears more than once or if they are better recognized than the full name (e.g., NASA, PG&amp;E). Similarly, avoid using in headlines or page headers unless well recognized.</p><p>Exception: Acronyms and initialisms may be used for CMEI News and blog items distributed via GovDelivery. Headlines should be written to be relatable and easy to understand.</p><p>Common abbreviations/acronyms/initialisms that do not require spelling:</p><ul><li>no. (number)</li><li>vs. (versus)</li><li>ASAP (as soon as possible)</li><li>R&amp;D (research and development)</li><li>centimeter (cm)</li><li>meter (m)</li><li>foot (ft.)</li><li>pound (lb.)</li></ul><p><strong>Web content.&nbsp;</strong>The above guidelines apply to web content as well, but try to avoid abbreviations/acronyms/initialisms, especially in headers and webpage or article titles, and use a shortened version of the name on second reference. If you can’t avoid, define it the first time used on each webpage.</p><p><strong>Technical measurement units.</strong> Spell out a technical abbreviation in full in text when used without numerals.</p><p>Abbreviate units of measurement when they are used with a numeral or numeric value. With a few exceptions (such as %, °, $, &lt;, and ¢), use a space to separate them from numerals.</p><p><em>In text without numeral:</em> “a few centimeters” not “a few cm.”</p><p><em>In text with numeral:</em></p><ul><li>15 cm</li><li>15 m</li><li>15 dB</li><li>15%</li><li>$15</li><li>&lt;15 W</li></ul><p><strong>Pluralization.</strong> Use a small s (no apostrophe) for plurals of most abbreviations. For plurals of units of measurement, omit the s.</p><ul><li>PUCs not PUC’s</li><li>CFCs, not CFC’s</li><li>15 cm, not 15 cms</li><li>6 m, not 6 ms</li><li>75 dB, not 75 dBs</li><li>40 W, not 40 Ws</li></ul><p><strong>Possessiveness.</strong> If the possessive is used with a name or term, the apostrophe should go after the spelled-out name or term and also with the acronym/abbreviation/initialism. However, preference is to write in a way that avoids use of the possessive.</p><blockquote><p>The U.S. Department of Energy’s (DOE's) Office of Critical Minerals and Energy Innovation (CMEI) is in charge of the program.</p></blockquote><p><strong>Abbreviating equations and references.</strong> You can abbreviate “equation” and “reference” when used with numbers. Spell them out at the beginning of a sentence.</p><blockquote><p>See Eq. 1-1, Eq. 2-7, and Ref. 10.</p><p>Equation 2-1 shows the relation.</p></blockquote><p><strong>Abbreviating in journals.</strong> For a journal article, consult the publisher’s or professional society’s guidelines for abbreviations, if available.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--abstract">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--abstract" id="abstract">
        
        abstract
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="abstract"></a>An abstract usually accompanies a journal article or conference paper. It is an important indexing and research tool. An abstract is usually 200 to 250 words in length and the content summarizes and highlights the major points of the journal article, conference paper, or report. It may also include a brief description of the purpose, scope, and methods used to reach the conclusions.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--academic-degrees">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--academic-degrees" id="academic">
        
        academic degrees
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="academic"></a>Avoid the use of academic degrees unless it’s absolutely necessary to establish credentials.&nbsp;</p><p>When writing out a degree, do not use an acronym unless it is used multiple times on a page, or if it is being used at the end of a <a href="#capitalization">person's title</a>. Acronyms for level of education (and certifications) should appear after a name and set it off with commas. Use them only on first reference.&nbsp;</p><p>Use periods between acronym letters, e.g., Ph.D., B.A., M.A., and LL.D. (exception: MBA).&nbsp;</p><p>The only academic title to appear before a name is "Dr." and that should only be used for a person who has received a medical degree at the doctoral level, not doctorates in philosophy, for example, per AP style.</p><p>Do not capitalize level of degree or subject when written out. Also, use an apostrophe in bachelor’s, master’s degree, etc. Neither Bachelor of Arts nor Master of Science includes a possessive apostrophe.&nbsp;</p><blockquote><p><em>She has a master’s degree in engineering.</em><br><em>He has a Bachelor of Arts from Penn State University.</em><br><em>Jane Doe, Ph.D., spoke at the conference.</em><br><em>Dr. John Smith, D.V.M., received his doctorate in veterinary medicine from the University of Somewhere in 2012.</em></p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--acknowledgments">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--acknowledgments" id="acknowledgments">
        
        acknowledgments
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="acknowledgments"></a>Acknowledge the reviews, funding, and other assistance of individuals and groups in CMEI technical reports and papers. Acknowledgments can go in a preface or foreword in a technical report, or they can be on a separate page if they are extensive. They also often follow the main text in professional journal articles. Journals usually contain examples or instructions for authors.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--acronyms">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--acronyms" id="acronyms">
        
        acronyms
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="acronyms"></a>See <a href="#abbreviations">abbreviations, acronyms, and initialisms</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--addresses">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--addresses" id="addresses">
        
        addresses
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="addresses"></a>Use U.S. Postal Service abbreviations (CO for Colorado and DC for District of Columbia) for states listed in bibliographies, references, and full addresses (when they include streets or post office boxes).</p>

<blockquote>
<p><br>
P.O. Box 123<br>
Denver, CO 80101</p>
</blockquote>

<p>In text, when referring to a state with a city or by itself, spell out the state name in full, except for the District of Columbia (D.C.).</p>

<blockquote>
<p>The state energy office is stepping up solar retrofit activities in Massachusetts.</p>
</blockquote>

<p>See also <a href="#states">states and countries</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--air-conditioning">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--air-conditioning" id="air-conditioning">
        
        air conditioning
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="air-conditioning"></a>Air conditioning is two words when used as a noun and hyphenated when used as an adjective.</p>

<blockquote>
<p>Air conditioning is energy intensive.<br>
The efficiency of the air-conditioning system can be improved.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--american-recovery-and-reinvestment-act-of-2009-arra">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--american-recovery-and-reinvestment-act-of-2009-arra" id="recovery-act">
        
        American Recovery and Reinvestment Act of 2009, ARRA
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="recovery-act"></a>Spell out the full name on first reference, use “Recovery Act” in subsequent references instead of “ARRA.” Do not put “Recovery Act” in parentheses.</p><p><em>Correct</em></p><blockquote><p>The American Recovery and Reinvestment Act of 2009 was designed to spur economic growth while creating new jobs and saving existing ones. Through the Recovery Act, DOE provided billions to support a wide range of renewable energy projects.</p></blockquote><p><em>Incorrect</em></p><blockquote><p>The American Recovery and Reinvestment Act of 2009 (ARRA)</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--ampersand-">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--ampersand-" id="ampersand">
        
        ampersand (&amp;)
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="ampersand"></a>Ampersands should be used with acronyms and initialisms (e.g., R&amp;D), left navigation menus, right navigation menus, and a website’s top banner (but not the heading). Ampersands are used in an official company or initiative names. Do not use “&amp;” to mean “and” in other situations.</p>

<p><em>Correct</em></p>

<ul>
	<li>PG&amp;E</li>
	<li>Leadership in Energy &amp; Environmental Design</li>
	<li>R&amp;D</li>
</ul>

<p><em>Incorrect</em></p>

<ul>
	<li>R and D</li>
	<li>Research &amp; Development</li>
</ul></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--appendices-or-appendixes">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--appendices-or-appendixes" id="appendices">
        
        appendices or appendixes
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="appendices"></a>Include detailed background or technical information in one or more appendices (preferred spelling, but either is acceptable). Large, detailed tables belong in an appendix. If more than one appendix, title each with letters (Appendix A, B, C, etc.) and name figures and tables to correspond to the title (Figure A-1, Table B-2, etc.). If only one appendix, title it "Appendix" rather than "Appendix A." If only one appendix exists, tables and figures should still be labeled with a letter and number (Figure A-1, Table A-1).</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--assure-ensure-and-insure">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--assure-ensure-and-insure" id="assure">
        
        assure, ensure, and insure
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><strong><a id="assure"></a>Assure </strong>means to give confidence. <strong>Ensure </strong>means to make certain. <strong>Insure </strong>means to obtain insurance.</p>

<blockquote>
<p>The manufacturer assured the group the equipment would work properly.</p>

<p>Ensure the lid is fitted properly before starting the experiment.</p>

<p>The laboratory must insure the new equipment before it can be used.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--author-date-citations">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--author-date-citations" id="author-date">
        
        author-date citations
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="author-date"></a>Following is the preferred style for CMEI reports and papers. Do not use a comma between the author’s last name and the year. See <a href="#references">references and citations</a>.</p><blockquote><p>The critical mineral supply chain impacts the energy industry, government policymaking, and society (Holdren 2001; Kalicki and Goldwyn 2005).</p></blockquote></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="b">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--b">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--b" id="b">
        
        B
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="B"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--balance-of-systems-bos">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--balance-of-systems-bos" id="bos">
        
        balance of systems, BOS
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="bos"></a>Balance of systems (not system) represents all components and costs other than the photovoltaic modules/array. It includes design costs, land, site preparation, system installation, support structures, power conditioning, operation and maintenance costs, indirect storage, and related costs.</p>

<blockquote>
<p>Whether you decide to connect your home renewable energy system to the electric grid or not, you will need to invest in some additional equipment (called "balance of systems") to condition the electricity, safely transmit the electricity to the load that will use it, and/or store the electricity for future use.</p>

<p>Several analyses have examined non-module PV system hardware costs, including the costs of power electronics and other BOS hardware elements.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--bandgap">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--bandgap" id="bandgap">
        
        bandgap
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="bandgap"></a>Bandgap (one word, noun or adjective) is a property that signifies the minimum amount of energy needed to free electrons so that the semiconductor material in a solar cell can conduct electricity. The bandgap determines the amount and type of light a semiconductor absorbs in a solar cell.</p><blockquote><p>NLR’s research shows that Ga2 O3 has the potential to have a 3–5X higher bandgap, as well as a 2–3X higher operating temperature compared to silicon.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--baseload">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--baseload" id="baseload">
        
        baseload
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="baseload"></a>The average amount of electric power that a utility must supply in any period. Baseload refers to the minimal load that is always there.<br>
Do not use this term in reference to a generation source, as this usage is outdated and not consistent with how the modern electricity system works.</p>

<p>Instead, focus on the attributes of resources and the services they provide to the system.</p>

<blockquote>
<p>Flexible CHP systems are an on-site electricity generation resource that efficiently provide both an electrical baseload and surplus.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--bibliography">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--bibliography" id="bibliography">
        
        bibliography
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="bibliography"></a>A bibliography is a list of works that are related to your subject or publication but not cited, either by author or by number, in text. Alphabetize works in bibliographies according to the last name of the first author. Some bibliographies are titled "For Further Reading." Compile in-text citations of literature and other sources in a list of references. See <a href="#references">references and citations</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--bioeconomy">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--bioeconomy" id="bioeconomy">
        
        bioeconomy
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="bioeconomy"></a>A global transition to efficient use of energy from renewable biomass resources.</p><blockquote><p>An analysis reported in <em>Nature </em>concluded that U.S. revenues from the bioeconomy amounted to more than $324 billion in 2012.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--bioenergy">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--bioenergy" id="bioenergy">
        
        bioenergy
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="bioenergy"></a>Energy produced from biomass, including biofuels and biopower.</p>

<blockquote>
<p>Traditional focuses on biofuels and bioenergy have been expanded to include the energy and carbon efficiency of biomass conversion, as well as bioproducts, biochemicals, and biomaterials.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--bioenergy-feedstocks">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--bioenergy-feedstocks" id="bioenergy-feedstocks">
        
        bioenergy feedstocks
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="bioenergy-feedstocks"></a>Biomass and waste materials that can be used to produce biofuels, bioproducts, and biopower. The raw materials undergo preprocessing operations (baling, grinding, leaching, blending, pelleting, or packaging) to meet quality characteristics for conversion at a biorefinery.</p>

<blockquote>
<p>Examples of bioenergy feedstocks include corn stover, switchgrass, and woody biomass.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--biofuels">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--biofuels" id="biofuels">
        
        biofuels
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="biofuels"></a>Liquid or gaseous fuels such as ethanol, methanol, methane, and hydrogen that are produced from biomass feedstocks. "Drop-in"&nbsp;biofuels are liquid or gaseous fuels that are compatible with existing vehicles and fuel distribution and storage infrastructure, such as renewable natural gas, renewable gasoline, renewable diesel, and renewable jet fuel.</p>

<blockquote>
<p>This work will help researchers realize the potential of an algal biofuel industry capable of producing billions of gallons per year of renewable diesel, gasoline, and jet fuels.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--biomass">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--biomass" id="biomass">
        
        biomass
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="biomass"></a>An energy resource derived from plant- and algal-based materials or organic waste streams. It includes agricultural residues, forest resources, perennial grasses, woody energy crops, algae, wet wastes (e.g., biosolids), sorted municipal solid waste, urban wood waste, food waste, biogas, and other waste streams. Biomass and waste are valuable energy resources as they can be converted to fuels, chemicals, or power.</p>

<blockquote>
<p>Idaho National Laboratory’s (INL) Bioenergy Feedstock Library is a biomass repository and research tool that contains information about the chemical, physical, and conversion performance properties of more than 90 crop types and factions from across the United States.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--biopower">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--biopower" id="biopower">
        
        biopower
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="biopower"></a>Electric power or heat derived from bioenergy feedstocks through direct combustion of the feedstock, through gasification and then combustion of the resultant gas, or through other thermal conversion processes. Power is generated with engines, turbines, or other equipment. Biopower technologies convert renewable biomass fuels into heat and electricity using processes similar to those used with fossil fuels.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--bioproducts">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--bioproducts" id="bioproducts">
        
        bioproducts
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="bioproducts"></a>Products, materials, or chemicals derived from bioenergy feedstocks. Examples include ethanol, plastics, polymers, and formic acid.</p>

<blockquote>
<p>Scientists engineering microbes for renewable fuels and bioproducts have developed a fast, efficient way to identify the most promising varieties.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--biorefinery">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--biorefinery" id="biorefinery">
        
        biorefinery
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="biorefinery"></a>A facility that converts biomass or waste resources into biofuels, bioproducts, and biopower. The biorefinery concept is analogous to a petroleum refinery, where a slate of multiple fuels and products are produced from a petroleum feedstock.</p>

<blockquote>
<p>The biorefinery in Emmetsburg, Iowa is the nation’s first commercial-scale cellulosic ethanol plant to use corn waste as a feedstock.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--british-thermal-unit-btu">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--british-thermal-unit-btu" id="btu">
        
        British thermal unit, BTU
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="btu"></a>The abbreviation for "British thermal unit" is Btu. Btu is used for both singular and plural cases and should be initial-capped.</p>

<blockquote>
<p>A requirement for significant reduction in the maximum Btu input rate of decorative vented gas fireplaces would impose substantial burdens on manufacturers.</p>

<p>Estimates show that the energy savings in the United States due to light-duty plug-in electric vehicles in 2019 was 44.8 trillion Btu, up 47% from 2018.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--bullets">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--bullets" id="bullets">
        
        bullets
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="bullets"></a>A bulleted list requires at least two items.</p><p><strong>Make bulleted lists parallel in construction</strong> (begin all the items in the list with the same part of speech, such as a verb or a noun).</p><ul><li>Make sure items are either all phrases or all complete sentences.</li><li>Punctuate all items consistently.</li></ul><p><strong>Punctuating bulleted lists.</strong> Begin each item with a capital letter and end each item with a period or other terminal punctuation (e.g., !, ?), except when the bulleted items consist of short noun phrases or single words. In such cases, punctuate only the final bullet.</p><p>Generally, CMEI prefers AP Style over Chicago Style on punctuating bulleted lists, except when AP Style end-punctuates each short phrase or single-word bullet. CMEI makes an exception to this, preferring that short phrases and single-word bullets omit end punctuation, except for the final bullet. See <a href="#lists">lists</a>.</p><p><em>Short phrases or single-word list punctuation</em></p><blockquote><p>FOAs issued in the last three years:</p><ul><li>DE-FOA-0002168</li><li>DE-FOA-0002032</li><li>DE-FOA-0001847<em>.</em></li></ul></blockquote><p><em>Complete sentence punctuation</em></p><blockquote><p>Use bulleted lists sparingly to:</p><ul><li>Highlight important items.</li><li>Draw attention to main points.</li><li>Help readers find information.</li></ul></blockquote><p>Do not use commas or semicolons at the end of bullets.</p><p><em>Incorrect</em></p><blockquote><p>In response to the President’s executive order, we are:</p><ul><li>Investing in supply-chain research and development,</li><li>Soliciting advice from private industry, and</li><li>Accelerating the development of new energy-storage technologies.</li></ul></blockquote><p>Use numbered or lettered lists instead of bullets if you want to refer to items in a list or procedure elsewhere in the text.</p><blockquote><p>Steps needed to execute the order include:</p><ol><li>Request the form.</li><li>Identify the target.</li><li>Complete the form, including all boxes.</li><li>Obtain PA approval of request.</li></ol><p>Concerning Step 2, consult the database to identify the target accurately, including all descriptors.</p></blockquote><p><strong>Formatting</strong>. In text, the first level of bullet is indented 0.25 in., and text begins at the 0.5-in. mark. Each subsequent level of bullet is sequentially indented 0.5 in. In lists of items that are more than one line, each bulleted item is followed by a 6-pt. space.</p><p><strong>Web content.</strong> Except for the indenting and spacing formats for reports, all the above guidelines apply.</p><p>A Cascading Style Sheet (CSS) governs web content on energy.gov. Web coordinators should check that the bulleted paragraph type they use has the correct spacing, as the energy.gov style does allow some flexibility. When formatting bullets on the website, there should be a space between the text above the bullets and the first bullet. To help facilitate scanning, consider a space between each bulleted item when the bulleted text is long.</p></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="c">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--c">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--c" id="c">
        
        C
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="C"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--capitalization">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--capitalization" id="capitalization">
        
        capitalization
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="capitalization"></a><strong>Proper names.</strong> These include government programs, official projects, formal groups, organizations, companies, titles when they precede a name (use lowercase in titles that follow the name), specific geographic areas or features, and ethnic groups.</p><p>For companies and products with stylized lowercase or “camel cap” names (e.g., iPod), use the company’s or product’s preferred capitalization.</p><blockquote><p><em>Sandia National Laboratories</em><br><em>American Wind Energy Association</em><br><em>President and Chief Executive Officer Christine Johnson; Christine Johnson, president and chief executive officer</em><br><em>the Southwest</em><br><em>Lake Powell</em><br><em>the Colorado River</em><br><em>African, Asian, Caucasian, Hispanic, or Native Americans</em></p></blockquote><p><em>Also capitalize names shortened from proper nouns per DOE style.</em></p><blockquote><p><em>National Lab</em><br><em>Administration</em><br>Department<br>Office</p></blockquote><p><strong>Taxonomic names.</strong> For botanical and zoological divisions, capitalize the names of all divisions higher than species: genera, families, orders, classes, and phyla. Put genera, species, and varieties in italics. Spell out at first mention and then abbreviate after.</p><blockquote><p><em>Clostridium thermocellum</em> (first mention)<br><em>C. thermocellum</em> (subsequent)<br><em>Escherichia coli</em> (first mention)<br><em>E. coli </em>(subsequent)</p></blockquote><p><strong>Headings, table titles, figure titles, and captions.</strong> Capitalize the main words of table titles and most headings and subheadings, including the second word in a hyphenated term. Verbs, including “is” and “are,” are always capitalized, including the word "to" in their infinitive form.</p><p>Do not capitalize articles (a, an, the); conjunctions (and, or, nor, but) less than five letters (e.g., for, of, to, from) unless they begin the title or heading.</p><p>Exception: If the heading takes up multiple lines, and the article or preposition comes at the beginning of a line, capitalize the article or preposition. (Applies mainly to printed products, as headlines on digital platforms may adjust automatically.)</p><blockquote><p><em>PV Program Five-Year Plan</em><br><em>Table 1. Number and Frequency of Defects in Six Samples</em><br><em>Development of Method To Detect Anomalies</em><br><em>U.S. Department of Energy Announces Tribal Grant Relief Due to COVID-19</em><br><em>Even Equal Pay Day Is Not Equal</em><br><em>Prize Helps To Advance Awareness About Electronic Waste</em></p></blockquote><p>Capitalize only the first word and proper nouns in figure captions.</p><blockquote><p><em>Figure 1. Results for the electrochromic window</em><br><em>Results for E. coli</em></p></blockquote><p>Follow the style recommended by your professional society or journal publisher regarding “figure” and its abbreviation (Fig.) when you prepare a paper or an article for submission. Many societies and publishers recommend lowercasing everything but the first word and proper nouns in table titles, subheads, and captions.</p><p><strong>States.</strong> Capitalize the names of states but capitalize "state" only when it appears with an official name, such as Kansas State Legislature. On second reference, state is never capitalized; however, Legislature is.</p><p>Capitalize "legislature"&nbsp;when preceded by the name of a state: the Kansas Legislature. Retain capitalization when the state name is dropped but the reference is specifically to that state’s legislature.</p><blockquote><p><em>the state of Colorado</em><br><em>Washington state</em><br><em>The forecast does not apply to the states below the Mason-Dixon line.</em><br><em>The Pennsylvania State University (Penn State)</em><br><em>The Washington State Legislature is winding up; both houses of the Legislature adjourned today.</em></p></blockquote><p><strong>Titles (professional)</strong>. Capitalize titles when they precede the person’s name. Lowercase titles and names of groups when they follow the name.</p><p>Former or future titles are also capitalized when preceding the person’s name. Except for “Acting,” which CMEI prefers to capitalize, lowercase qualifiers (e.g., former, deposed, presumed).</p><p>See <a href="#academic">academic degrees</a> to learn more about tiles for education levels.</p><blockquote><p><em>The Secretary of Energy was appointed in 2017.</em><br><em>Secretary of Energy John Doe attended the conference.</em><br><em>John Doe, the current secretary of energy, previously served as the deputy secretary of energy.</em><br><em>Chief Operating Officer Mark Wilson</em><br><em>Mark Wilson, the chief operating officer</em><br><em>former Secretary of State Henry Kissinger</em><br><em>Acting Chief of Police Robert Hand</em><br><em>John Doe, Ph.D., received his doctoral degree in physics in 2012.</em><br><em>Dr. Jane Smith, M.D., received her medical degree from University of Somewhere in 2012 after obtaining a master's degree in chemistry.</em></p></blockquote><p><strong>Titles (composition).</strong> Titles of books, movies, plays, poems, albums, songs, operas, radio and television programs, lectures, speeches, and works of art. Capitalize all words in a title except articles (a, an, the); prepositions of four or fewer letters (for, of, on, up); and conjunctions of four or fewer letters (and, but, for, nor, or, so, yet) unless any of those start or end the title.</p><blockquote><p><em>Gone with the Wind</em><br><em>Applied Physics Letters</em><br><em>The Denver Post</em><br><em>Science</em></p></blockquote><p><strong>Trade names.</strong> Commonly used government-owned trademarks should use the trademark notice symbols. Include the symbol in first use in heading and in body text; thereafter, omit the symbol.</p><p>Capitalize trade or brand names for third-party trade names (Kleenex, Pyrex). Do not use trademark notice symbols with third-party products or services.</p><p>Refer to the company’s literature or stationery if you’re not sure. See also the online database of <a href="https://www.uspto.gov/trademarks">current trademarks</a>.</p><p><em>Common government-owned names and symbols</em></p><blockquote><p>BUILD4SCALE™<br>Energy Earthshots™<br>EnergyPlus®<br>ENERGY STAR®<br>H2@SCALE™<br>Home Energy Score™<br>HydroGEN Advancing Water Splitting Materials™<br>RACE TO ZERO™<br>RAPID TOOLKIT®<br>Solar Decathlon®<br>SOLSMART®</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--captions">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--captions" id="captions">
        
        captions
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="captions"></a>Accompany all substantive photos, figures, and images with a caption. Begin captions with a capitalized word and use lowercase thereafter, except for proper nouns and capitalized abbreviations.</p><blockquote><p>General operation of electrolysis process for water splitting for a proton-conducting system<br>A simplified schematic of the Drake Landing Solar Community in Canada</p></blockquote><p>Don’t include a period at the end of a caption unless it’s a complete sentence or there is a subcaption. A complete sentence with a verb describing the action in the photo is preferred.</p><p>If one or more full sentences follow the incomplete sentence (as a continuing caption or subcaption), each should have a period. Don’t bold subcaptions.</p><blockquote><p><strong>Figure 2-1. Photoconductivity spectra of a composite CIS thin film.</strong> Inset: The probable energy band diagram.</p></blockquote><p>Captions that accompany a photograph should credit the photographer, including name, organization, and year, if available. The credit should be italicized and does not include a period.</p><blockquote><p>Parabolic trough collectors concentrate the sun’s energy on an oil-filled tube running along the focal line of the parabolic trough. <em>Photo by Randy Montoya, Sandia Labs</em></p></blockquote><p>Follow <a href="/eere/communicationstandards/eere-publication-exhibit-template-and-logo-standards-and-guidelines">CMEI Communications Standards and Guidelines</a> regarding placement, formatting, and font use in figures and captions.</p><p>See <a href="#figures">figures</a> and <a href="#photographs">photographs</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--chemical-terms">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--chemical-terms" id="chemical">
        
        chemical terms
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="chemical"></a>Do not use a hyphen in most chemical expressions, even when the terms are used as modifiers. Use a hyphen after prefixes when it is the standard for certain chemical formulas. Use a hyphen to indicate mixtures or combinations.</p>

<p><em>No hyphen</em></p>

<blockquote>
<p>carbon dioxide levels<br>
hydrogen ion activity</p>
</blockquote>

<p><em>Standard</em></p>

<blockquote>
<p>L(+)-2, 3-butanediol<br>
trans-glycol</p>
</blockquote>

<p><em>Mixtures or combinations</em></p>

<blockquote>
<p>hexane-benzene</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--citations">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--citations" id="citations">
        
        citations
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="citations"></a>See <a href="#references">references and citations</a> for guidance on author-date and numbered citations.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--close-spaced-sublimation">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--close-spaced-sublimation" id="close-spaced">
        
        close-spaced sublimation
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="close-spaced"></a>The term is not "closed-space sublimation" nor "close-space sublimation."</p>

<blockquote>
<p>The research determined whether it was possible to produce an open-circuit voltage (Voc) greater than 1.1V using close-spaced sublimation.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--cogeneration">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--cogeneration" id="cogeneration">
        
        cogeneration
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="cogeneration"></a>Do not use a hyphen.</p>

<blockquote>
<p>Conventional large cogeneration systems are well developed, widely deployed, and utilize readily available thermal technologies.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--colons">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--colons" id="colons">
        
        colons
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="colons"></a>Colons formally introduce a numbered or bulleted list, series, question, or amplification. They can also separate the parts of a ratio.</p><p>Capitalize the first word after a colon only if it is a proper noun or the start of a complete sentence.</p><blockquote><p>We test three types of collectors: flat plates, evacuated tubes, and parabolic troughs.<br>We test three types of collectors:</p><ul><li>Flat plates</li><li>Evacuated tubes</li><li>Parabolic troughs.</li></ul><p>We added enough water to obtain a 3:1 dilution.</p></blockquote><p>Use a colon when a noun (such as “the following”) introduces a list in text.</p><blockquote><p>The electric efficiencies, thermal output, and potential overall efficiency are shown in the following chart:</p></blockquote><p>Don’t use a colon after a verb or preposition that precedes or introduces a list (are, includes, to, with, between, etc.).</p><blockquote><p>The research areas include exploration and characterization, subsurface accessibility, and subsurface enhancement.</p></blockquote><p>Commas, not colons, usually follow words such as "that is," "namely,"&nbsp;or "such as."</p><blockquote><p>Reducing the cost of cement materials in geothermal well construction can follow a similar path to that of reducing casing; that is, using less material, developing alternative approaches, and investigating leaner casing designs.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--commas">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--commas" id="commas">
        
        commas
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="commas"></a>Use a comma to separate items in a series, including the next-to-last word in the series (also known as the serial comma). CMEI prefers Chicago Style over AP Style on this issue.</p><blockquote><p>We develop solar thermal, wind, biomass, and photovoltaic energy technologies.</p></blockquote><p>Comma placement is key in resolving ambiguity. Consider: "We will strengthen U.S. leadership by developing public-private partnerships, technology and interconnection standards, and the dissemination of high-quality market data."&nbsp;Should a comma follow "technology"? Does it mean "technology standards"&nbsp;and "interconnection standards"? Or are "technology and interconnection standards"&nbsp;one entity?</p><p>Use a comma to separate the parts of a compound sentence linked by a coordinating conjunction (and, but, or, nor) when each part has its own subject and verb (unless they’re very short).</p><blockquote><p>I laughed at the unintentional joke, but she frowned.</p></blockquote><p>Use commas to set off nonessential or nonrestrictive words, phrases, and clauses from the rest of the sentence. The commas signal that the information between them is something extra and not essential to the sentence meaning.</p><blockquote><p>The subsystem, which takes a day to install, will be delivered in two weeks.</p></blockquote><p>Do not use commas to set off restrictive words or phrases that are essential to the sentence meaning.</p><blockquote><p>Only the sensors that were attached to the outer edge failed. (essential phrase)<br>The system will work efficiently only if it includes storage. (essential phrase)</p></blockquote><p>See also <a href="#which">which and that</a>.</p><p>Use commas to enclose the state name when it follows a city and the year when it follows the month and day.</p><blockquote><p>The test systems in Gardner, Massachusetts, are performing well.<br>The next test sites will be in Golden, Colorado, and Chapel Hill, North Carolina.<br>On April 11, 2019, the committee members completed five of the six objectives.</p></blockquote><p>Do not use a comma to separate compound subjects or compound verbs.</p><blockquote><p>Theorists and nonspecialists alike agree on the importance of the discovery. (No comma between the two parts of the compound subject.)<br>The researchers rolled out the thin metal sheet and formed it into coils. (No comma between the two parts of the compound verb.)</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--compose-and-comprise">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--compose-and-comprise" id="compose">
        
        compose and comprise
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="compose"></a>"Composed of" is correct; "comprised of" is incorrect.</p>

<blockquote>
<p>The United States is composed of 50 states.<br>
The department comprises four groups; each group is composed of five to seven scientists, technicians, and support staff.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--comprise-and-compose">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--comprise-and-compose" id="comprise">
        
        comprise and compose
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="comprise"></a>See <a href="#compose">compose and comprise</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--compound-words-and-modifiers">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--compound-words-and-modifiers" id="compound">
        
        compound words and modifiers
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="compound"></a><strong>Verb phrases (verb, noun, and adjective forms).</strong> Verb phrases that contain an adverb (build <em>up</em>, set <em>up</em>, start <em>up</em>, break <em>down</em>) are usually written as two words. The noun and adjective forms of these words are either one word (no hyphen) or a hyphenated form of the words. However, there are exceptions. Refer to the dictionary for the correct spelling.</p><blockquote><p>We observed the slow buildup of biofouling on the blades.<br>We helped with the setup.<br>The startup costs were higher than we estimated.<br>I think I’m having another breakdown.</p></blockquote><p><strong>Compound modifiers with and without hyphens.</strong> Use a hyphen to indicate that words have been combined into a compound modifier—a descriptive expression of two or more words that form one new meaning. For example, in the term “flat-plate collector,” “flat-plate” is the compound modifier.</p><blockquote><p>low-level radiation<br>last-minute addition<br>fatigue-induced wear<br>five-year plan<br>nine-story building<br>ground-source heat pump</p></blockquote><p>Although modern writing often eliminates hyphens, they help prevent ambiguity. To see how adding the hyphen can prevent confusion, consider: “The scientists tested a new defect causing gas,” versus “The scientists tested a new defect-causing gas.” In the first example, the scientists might seem to have been testing a defect; in the second example, it’s clear that they have tested a gas.</p><p>See <a href="#hyphens">hyphens</a> to learn more about what terms need no hyphens.</p><p><strong>Compound words containing prefixes and suffixes</strong>. No hyphen is needed between many prefixes and suffixes and the root words, unless the root word is a proper noun.</p><p>Use a hyphen if the word that follows is a proper noun, or is capitalized, e.g., pre-Civil War.</p><p>Use two hyphens to join double prefixes, e.g., sub-sub-paragraph.</p><p>Prefixes that usually require a hyphen include "ex," "self," and "quasi." See <a href="#hyphens">hyphens</a>.</p><p>Compound words containing prefixes and suffixes that don’t need a hyphen:</p><blockquote><p>multidimensional<br>multiyear<br>reevaluated<br>reuse<br>threefold, hundredfold (but 100-fold)<br>subassembly<br>prescreening<br>postdoctoral<br>retroactive<br>nonspecialist</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--concentrating-photovoltaics-cpv">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--concentrating-photovoltaics-cpv" id="cpv">
        
        concentrating photovoltaics, CPV
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="cpv"></a>A solar technology that uses lenses or mirrors to concentrate sunlight onto high-efficiency solar cells.</p>

<blockquote>
<p>He designs custom data acquisition systems for a wide variety of concentrating photovoltaics (CPV) and flat-plate PV projects.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--concentrating-solar-thermal-power-csp">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--concentrating-solar-thermal-power-csp" id="csp">
        
        concentrating solar-thermal power, CSP
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="csp"></a>A solar technology that uses mirrors to reflect and concentrate sunlight onto receivers that convert solar energy to heat. This thermal energy is then used to produce electricity with a steam turbine or heat engine driving a generator.</p>

<blockquote>
<p>This project will seek to integrate multiple thermochemical energy storage components into a CSP design that enables a plant to have multiple storage durations, including daily and long-term.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--congress-and-congressional">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--congress-and-congressional" id="congress">
        
        Congress and congressional
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="congress"></a>Capitalize "U.S. Congress" and "Congress" when referring to the U.S. Senate and House of Representatives. Lowercase "congressional" unless it is part of a proper name.</p>

<blockquote>
<p>The U.S. Congress is reviewing congressional salaries. A full list is available in the <em>Congressional Record</em>.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--consortia-consortium">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--consortia-consortium" id="consortia">
        
        consortia, consortium
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="consortia"></a>Consortia is the plural of consortium (two or more consortia).</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--cooperative-research-and-development-agreement-crada">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--cooperative-research-and-development-agreement-crada" id="crada">
        
        cooperative research and development agreement, CRADA
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="crada"></a>On first reference, use lowercase for "cooperative research and development agreement" because it's not a proper noun. On second reference, use the acronym "CRADA."</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--covid-19-coronavirus-disease-2019">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--covid-19-coronavirus-disease-2019" id="covid-19">
        
        COVID-19, coronavirus disease 2019
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="covid-19"></a>On February 11, 2020, the World Health Organization announced an official name for the disease that caused the 2019 novel coronavirus outbreak. The new name of this disease is coronavirus disease 2019, abbreviated as COVID-19. CMEI preference is to follow the style established by the CDC. See the <a href="/covid/doe-safety-strategy">Department of Energy COVID-19 Workplace Safety Plan</a>.</p><blockquote><p>The DOE COVID-19 Coordination Team is responsible for establishing, implementing, and monitoring compliance with the health and safety requirements specified in this plan.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--coproduction">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--coproduction" id="coproduction">
        
        coproduction
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="coproduction"></a>A method of generating two useful products simultaneously. It often refers to combined-heat-and-power systems that produce both electricity and heat (e.g., steam). It also can refer to the joint production of electricity and hydrogen.</p>

<blockquote>
<p>The renewable electrolysis platform integrates renewable generation with hydrogen electrolyzers and storage infrastructure to help utilities and developers study the coproduction of electricity and hydrogen.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--countries-and-states">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--countries-and-states" id="countries">
        
        countries and states
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="countries"></a>See <a href="#states">states and countries</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--country">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--country" id="country">
        
        country
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="country"></a>Always lowercase unless part of a proper noun.</p>

<blockquote>
<p>Gas prices increased across the country during the 1970s.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--criteria-criterion">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--criteria-criterion" id="criteria">
        
        criteria, criterion
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="criteria"></a>"Criterion" is a singular noun (one criterion), and "criteria" is the plural (two or more criteria).</p>

<blockquote>
<p>While evaluation criteria provide comparability and adherence to stated objectives, the first criterion should be closely examined.</p>
</blockquote></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="d">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--d">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--d" id="d">
        
        D
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="D"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--dashes-em">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--dashes-em" id="dashes-em">
        
        dashes (em)
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="dashes-em"></a>Use dashes (often called “long dashes” or “em dashes”) to set off parenthetical (nonessential but often illustrative) information. Also use dashes to set off a list of items separated by commas. CMEI prefers no spaces around the dash.</p><blockquote><p>The polymer components of the cell walls—cellulose, hemicellulose, and lignin—provide the feedstocks for these chemicals.</p></blockquote><p>Use an em dash to signal that an important point is going to be made or that a change in the construction of the sentence follows.</p><blockquote><p>The presentation concluded with a discussion of the two project factors that concern contractors the most—cost and time.</p><p>The major omission in the project assessment was the delay caused by the circuit failures—everyone knew about it, but no one mentioned it to the reviewers.</p></blockquote><p>Typically, commas, colons, and semicolons can be used in place of dashes, but dashes add special emphasis.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--dashes-en">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--dashes-en" id="dashes-en">
        
        dashes (en)
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="dashes-en"></a>Use shorter "en dashes" (rather than a hyphen or em dash) to indicate a range or to substitute for the word "to." Note: AP Style does not use en dashes.</p>

<blockquote>
<p>25–45 cm2<br>
2–5 runs per hour<br>
See sections 3.1–3.6<br>
Jan. 16–Feb. 3, 2011<br>
pages 1–5</p>
</blockquote>

<p>In date spans, do not use "from" with an en dash.</p>

<p><em>Correct</em></p>

<blockquote>
<p>from Jan. 16 to Feb. 3; Jan. 16–Feb. 3</p>
</blockquote>

<p><em>Incorrect</em></p>

<blockquote>
<p>from Jan. 16–Feb. 3</p>
</blockquote>

<p>Do not use an en dash (or hyphen) to mean "and;" the word "between" is followed by the word "and" (not "to").</p>

<p><em>Correct</em></p>

<blockquote>
<p>between 25 and 30</p>
</blockquote>

<p><em>Incorrect</em></p>

<blockquote>
<p>between 25–30</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--data-datum-dataset">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--data-datum-dataset" id="data">
        
        data, datum, dataset
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="data"></a>"Data" is the plural of "datum." The word "data" typically takes singular verbs and pronouns when writing for general audiences and in data journalism contexts. In scientific and academic writing, plural verbs and pronouns are preferred. "Dataset" refers to a collection of data and should be written as one word.</p><p><em>Scientific and academic audiences</em></p><blockquote><p>Assorted data were collected and filed.<br>A datum escaped his count.<br>The dataset includes all measurements.&nbsp;</p></blockquote><p><em>General audiences</em></p><blockquote><p>The data is sound.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--data-in-tables">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--data-in-tables" id="data-tables">
        
        data in tables
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="data-tables"></a>Place a zero to the left of the decimal in any number less than 1 in text and tables (e.g., 0.5, 0.039). Align columns of numerals vertically on the decimals. When the units of measurement for the data are different, alignment is not necessary (but be sure to specify the units).</p>

<p>When the units of measurement for the data are different, alignment is not necessary (but be sure to specify the units).</p>

<p>&nbsp;</p>

<p>&nbsp;</p>

<p>&nbsp;</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--dates">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--dates" id="dates">
        
        dates
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="dates"></a>Use the month, numeral, and year for announcements and publicity that convey deadlines or events. Follow American date order convention: month, day, year. Days of the week are not necessary but, if included, should be written out in full.</p><blockquote><p>Sept. 20, 2021<br>The event is scheduled for Monday, Sept. 20, 2021, at 10 a.m. ET.</p></blockquote><p>When a month is used with a specific date, abbreviate only Jan., Feb., Aug., Sept., Oct., Nov., and Dec. Spell out when using alone or with a year alone.</p><blockquote><p>The webinar will be held Jan. 5, 2019.<br>The conference took place in January 2018.<br>The workshop series ended in January.</p></blockquote><p>Use common month abbreviations when a full date is provided. Use cardinal numbers for the day.</p><blockquote><p>Jan. 1, 2010<br>May 6, 1990</p></blockquote><p>When a phrase lists only a month and a year, do not separate the year with commas.</p><p><em>Correct</em></p><blockquote><p>September 2020</p></blockquote><p><em>Incorrect</em></p><blockquote><p>July, 2017</p></blockquote><p>When a phrase refers to a month, day, and year, set off the year with commas.</p><blockquote><p>Feb. 14, 2013, was the target date.</p></blockquote><p>Date ranges use the en dash with no spaces. Do not use ordinal numbers (2nd, 3rd, 4th) unless writing prepared remarks or a speech, e.g., "On January 1st, the president announced ...".</p><p><em>Correct</em></p><blockquote><p>Jan. 1–5, 2019</p></blockquote><p><em>Incorrect</em></p><blockquote><p>Jan. 1 to 5, 2019<br>January 1st through 5th, 2019</p></blockquote><p>&nbsp;We do not recommend including days of the week, but if necessary, consider separating that range from the date range.</p><blockquote><p><em>Monday through Thursday, Jan. 1–5, 2023.</em></p></blockquote><p>Decades do not include apostrophes unless preceded by "the."</p><blockquote><p>A woman in her 50s<br>the '90s; the mid-'90s<br>The 1860s saw the breakup of the Union.</p></blockquote><p>See <a href="#months">months and years</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--decision-maker">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--decision-maker" id="decision-maker">
        
        decision maker
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="decision-maker"></a>Decision maker is two words, but decision-making is hyphenated when used as a compound modifier. Note: AP Style hyphenates both decision-maker and decision-making.</p>

<blockquote>
<p>The PUC acts as the decision maker on the issue.<br>
decision-making process</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--degree-symbol">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--degree-symbol" id="degree">
        
        degree symbol
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="degree"></a>Place the degree symbol (º) next to the symbol for the temperature scale, with no space. Repeat the degree symbol in ranges. Express kelvins as K rather than as ºK; leave a space before the K.</p>

<blockquote>
<p>36ºC<br>
85ºF<br>
32º–36ºC<br>
85 K</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--department-of-energy">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--department-of-energy" id="doe">
        
        Department of Energy
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="doe"></a>See <a href="#usdoe">U.S. Department of Energy</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--desalination">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--desalination" id="desalination">
        
        desalination
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="desalination"></a>The process of removing salt from seawater.</p><blockquote><p>NLR scientists are actively researching desalination capabilities by partnering with technology leaders on several groundbreaking projects.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--disclaimer">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--disclaimer" id="disclaimer">
        
        disclaimer
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="disclaimer"></a>DOE requires that CMEI publications include a disclaimer. The disclaimer used depends on the type of publication. See <a href="/eere/communicationstandards/eere-branded-publications-developed-third-parties">CMEI-Branded Publications Developed by Third Parties</a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--dish-engine-systems">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--dish-engine-systems" id="dish">
        
        dish/engine systems
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="dish"></a>Use a slash rather than a hyphen. The dish/engine system is a concentrating solar power (CSP) technology with two major parts: the solar concentrator and the power conversion unit.</p>

<blockquote>
<p>Dish/engine systems use a parabolic dish of mirrors to direct and concentrate sunlight onto a central engine that produces electricity.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--dollars">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--dollars" id="dollars">
        
        dollars
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="dollars"></a>Express thousands of dollars using a comma.</p>

<blockquote>
<p>$5,000</p>
</blockquote>

<p>Express millions, billions, and trillions of dollars in decimal points, followed by the word. It is acceptable to use the abbreviation "M" or "B" in headlines.</p>

<blockquote>
<p>$3 million budget<br>
$1.2 billion<br>
DOE Announces $3M in Funding</p>
</blockquote>

<p>In technical reports and papers, use a dollar sign to express costs less than $1.</p>

<blockquote>
<p>$0.25<br>
$0.06 per kilowatt-hour</p>
</blockquote>

<p>No hyphens are used, even when the phrase is used as an adjective.</p>

<blockquote>
<p>Congress passed a $4.2 trillion budget just before midnight.</p>
</blockquote></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="e">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--e">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--e" id="e">
        
        E
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="E"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--eg-and-ie">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--eg-and-ie" id="eg">
        
        e.g. and i.e.
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="eg"></a>The abbreviation for the Latin phrase exempli gratia (e.g.) means "for example." The abbreviation for the Latin phrase id est (i.e.) means "that is." Both e.g. and i.e. are always followed by a comma. Do not italicize e.g. or i.e.</p>

<p>Use a semicolon before "i.e." and "e.g." and a comma after them when a clause (with a subject and verb) follows them.</p>

<blockquote>
<p>Making nanotechnology more scalable and manufacturable through new processes, e.g., nanotechnologies that can operate in room-temperature environments.</p>

<p>The team started with a recent development discovered during the conversion of methanol to fuels, i.e., the production of branched C4–C7 hydrocarbons while using beta zeolite catalysts.</p>

<p>Bring any two items; e.g., sleeping bags and tents are in short supply.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--electric-vehicle-ev">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--electric-vehicle-ev" id="ev">
        
        electric vehicle, EV
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="ev"></a>Electric vehicles, also known as plug-in electric vehicles, derive all or part of their power from electricity supplied by the electric grid.</p>

<blockquote>
<p>There are two basic types of EVs: all-electric vehicles (AEVs) and plug-in hybrid electric vehicles (PHEVs).</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--ellipses-ellipsis">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--ellipses-ellipsis" id="ellipses">
        
        ellipses, ellipsis
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="ellipses"></a>To shorten a quote, use ellipsis marks (three periods with a space on each side … or the ellipsis symbol) to indicate the omission.</p>

<p>If the words preceding the ellipsis form a grammatically correct sentence, put a period at the end and follow it by an ellipsis. Usually, ellipses aren’t needed at the beginning or end of quotes, just within them. To add a word(s) to the quote, enclose the added word or words in brackets to show it’s not part of the original quotation.</p>

<p>When quoting whole paragraphs but omitting text between any two of them, center three asterisks with spaces between them (* * *) between the paragraphs quoted. See also <a href="#quotation">quotation marks</a>.</p>

<blockquote>
<p>A participle is "a word having the characteristics of both verb and adjective ... [that] shows such verbal features as tense and voice. ..."</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--email">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--email" id="email">
        
        email
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="email"></a>Acceptable in all references for "electronic mail" without a hyphen between the "e" and mail. Use a hyphen with other e- terms: e-book, e-business, e-commerce.</p>

<blockquote>
<p>I sent an email to everyone involved with the project.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--energy-department">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--energy-department" id="energy-department">
        
        Energy Department
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="energy-department"></a>On first reference only, "Energy Department" may be used in communications to the press, including news story headlines. See <a href="#usdoe">U.S. Department of Energy</a>.</p><blockquote><p>Today, the Energy Department announced Round One winners of the Geothermal Manufacturing Prize.</p><p>Energy Department Announces Awardees of Funding Program</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--energy-star">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--energy-star" id="energy-star">
        
        ENERGY STAR
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="energy-star"></a>Always shown in all capitals, the first use of ENERGY STAR should include the registration mark (®), except when the first use occurs in a headline or header. In that case, include ® in the first use in the body. After first use, do not repeat the ®. There is no space between ENERGY STAR and the mark.</p>

<p><em>Headline</em></p>

<blockquote>
<p>What Is ENERGY STAR?</p>
</blockquote>

<p><em>Body Content</em></p>

<blockquote>
<p>ENERGY STAR® is the government-backed symbol for energy efficiency, providing simple, credible, and unbiased information that consumers and businesses rely on to make well-informed decisions.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--enhanced-geothermal-system-egs">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--enhanced-geothermal-system-egs" id="egs">
        
        enhanced geothermal system, EGS
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="egs"></a>While enhanced geothermal system (EGS) is preferred, it may also be referred to as "engineered geothermal system."</p>

<blockquote>
<p>A man-made enhanced geothermal system (EGS) can extract the abundant heat resource located tens of thousands of feet below the surface and put it to good use.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--equations">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--equations" id="equations">
        
        equations
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="equations"></a>All equation terminology must be defined and used consistently in text and in subsequent equations, figures, and tables.</p>

<blockquote>
<p>The conductive heat flow equation is:<br>
dQ/dt = AKdT/dx<br>
where<br>
dQ/dt = the time rate of heat transfer<br>
A = the area of an end contact<br>
K = the thermal conductivity<br>
dT/dx = the thermal gradient.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--ensure-insure-and-assure">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--ensure-insure-and-assure" id="ensure">
        
        ensure, insure, and assure
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="ensure"></a>See <a href="#assure">assure, ensure, and insure</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--et-al">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--et-al" id="et-al">
        
        et al.
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="et-al"></a>An abbreviation for the Latin phrase, et alia (neuter plural), meaning "and others." Most commonly, et al. indicates other contributors (authors, editors, etc.) in a reference list. Always follow et al. with a period; do not italicize.</p>

<blockquote>
<p>Feynman, Hawking, Sagan, et al.</p>

<p>The technical report, “New Approaches to Distributed PV Interconnection: Addressing Emerging Issues,” (McAllister et al. 2019) describes the emerging issues and various approaches being explored by the states.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--etc">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--etc" id="etc">
        
        etc.
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="etc"></a>Abbreviation for the Latin phrase, et cetera, meaning "and other things." Avoid "etc." if possible. Don’t add it to the end of a list beginning with "for example," or the abbreviation "e.g.," because each word in the list is an example, but "etc." is not. Etc. always ends in a period because it is an abbreviation. If "etc." is at the end of a sentence, it is always preceded by a comma.</p>

<blockquote>
<p>Letters, packages, etc., should go here.</p>

<p>Information needed includes compatible user interface, complete system wiring diagram (not just a one-line diagram), any needed black-box control devices, housings, cables, connectors, etc.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--e-terms">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--e-terms" id="e-terms">
        
        e-terms
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="e-terms"></a>Use a hyphen in terms that include "e" as an abbreviation for electronic, except for "email," which does not require a hyphen.</p>

<blockquote>
<p>e-book<br>
e-business<br>
e-commerce</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--executive-summary">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--executive-summary" id="executive">
        
        executive summary
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="executive"></a>An executive summary should appear before the table of contents. A summary is not usually needed in a short report. An executive summary can be as much as 5% to 10% of the document. It should stand independently of the full report, as they are frequently published as separate documents. Include a brief statement of the problem or proposal, background information, a concise analysis, and the main conclusions. Tables and figures should be numbered within the executive summary (i.e., Table ES-1, Figure ES-1).</p><p>See <a href="/cmei/communicationstandards/brand-guidelines-office-critical-minerals-and-energy-innovation#eere-template">CMEI Communications Standards</a>.</p></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="f">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--f">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--f" id="f">
        
        F
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="F"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--federal">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--federal" id="federal">
        
        federal
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="federal"></a>"Federal" should not be capitalized unless it’s part of a proper noun. Capitalize "federal" as part of formal names. Use lowercase to distinguish it from state, county, city, or local entities.</p>

<blockquote>
<p>Federal Trade Commission<br>
Federal Communications Commission<br>
federal government<br>
federal court<br>
federal judge</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--figures">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--figures" id="figures">
        
        figures
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="figures"></a>Figures include line drawings, graphs, charts, diagrams, schematics, flow charts, illustrations, and photographs. Number figures in a simple sequence (e.g., Figure 1, Figure 2). In long reports, papers, or book chapters, include section or chapter numbers in the figure numbers (e.g., Figure 1-1, Figure 1-2, Figure 2-1). All figures should include a figure title (minimum). A caption is optional but desired.</p>

<p>In print, use a consistent line weight in figures. Computer-generated figures must be clear so they can be reproduced easily. No period is needed after a caption if it is an incomplete sentence.</p></div>
      
            </div>

  

<div class="block block-layout-builder block-inline-blockmedia layout-align layout-center layout-full layout-width">
            
            
                                          <img src="/sites/default/files/styles/full_article_width/public/2021-08/eere_style-guide_final_apr21.jpg?itok=xba1wbcN" alt="Example line graph" typeof="foaf:Image" />

                          
              <div class="grid-container caption">
          <div class="grid-row">
            <div class="grid-col-1 caption-icon-wrapper">
                              <i class="fa-regular fa-camera"></i>
                          </div>
            <div class="grid-col-11">
              <div class="caption-text">
                                  Figure 1. Renewables as a percent of total installed capacity worldwide
                              </div>
              <div class="caption-attribution">
                              </div>
            </div>
          </div>
        </div>
                    
      </div>

<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p>If one or more full sentences follow the incomplete sentence (as a continuing caption or subcaption), each caption (including the opening incomplete sentence) should have a period. Don’t bold subcaptions.</p><blockquote><p>Figure 2-1. Photoconductivity spectra of a composite CIS thin film. Inset: the probable energy band diagram.</p></blockquote><p>See <a href="/eere/communicationstandards/eere-publication-exhibit-template-and-logo-standards-and-guidelines">CMEI Communications Standards and Guidelines</a> on placing and formatting figures, titles, and captions.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--fiscal-year-fy">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--fiscal-year-fy" id="fy">
        
        fiscal year, FY
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="fy"></a>Spell out "fiscal year" (e.g., Fiscal Year 2020) the first time; thereafter, abbreviate it using two capitals followed by a space before the full year. Always capitalize Fiscal Year when referring to a specific year.</p>

<p>The abbreviation, FY##, may be used to save space in charts and graphs. On websites, always spell out "fiscal year."</p>

<blockquote>
<p>Fiscal Year 2020<br>
FY 2001<br>
The accounting ended with the fiscal year.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--footnotes">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--footnotes" id="footnotes">
        
        footnotes
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="footnotes"></a>Use footnotes for explanatory or supplementary information. Preferably, place explanations, details, contradictions, and examples in the text rather than in footnotes.</p><p>In-text references may be used to cite others' works, but if sources are cited using footnotes, then a complete citation should be included (author, year, title, and hyperlink, at a minimum). Cited sources should be readily found if searched.</p><p>For all CMEI technical publications, citations/footnotes should be consistent, using the predominate style followed by the author in accordance with professional citation standards.</p><p>Use superscript numerals for footnote numbering. Mark the footnotes to tables in CMEI reports with superscript letters: <sup>a</sup>, <sup>b</sup>, <sup>c</sup>, etc. The footnote number follows all punctuation marks except the dash, which it precedes.</p><blockquote><p>We discussed these three stages of writing<sup>1</sup>—prewriting, writing, and revising.</p><p>Magnetocalorics, which have been used for cryogenic refrigeration since the 1930s, became an area of active research for room temperature applications after Pecharsky and Gschneidner (1997) first demonstrated the “giant magnetocaloric effect.”<sup>2</sup></p><p><sup>_____________________________</sup></p><p><sup>1</sup> Christa Riddle. 2015. "The 3 Steps of Successful Writing: Prewriting, Writing, and Rewriting." All About Writing Consulting. Accessed June 2019. https://allaboutwritingconsulting.com/2015/03/the-3-steps-of-successful-writing-prewriting-writing-rewriting.</p><p><sup>2</sup> The larger temperature differences are required for commercial viability of near room-temperature refrigeration.</p></blockquote><p>For more guidance on footnote style: <a href="https://www.easybib.com/guides/citation-guides/chicago-turabian/footnotes/">https://www.easybib.com/guides/citation-guides/chicago-turabian/footnotes/</a></p><p>See <a href="#references">references and citations</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--fractions">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--fractions" id="fractions">
        
        fractions
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="fractions"></a>Use words instead of numerals for simple fractions in text. See also <a href="#equations">equations</a>.</p>

<blockquote>
<p>a third of the way<br>
one-fifth its actual size<br>
three-fourths of the participants</p>
</blockquote>

<p>Write out complex fractions with numerals separated by a slash.</p>

<blockquote>
<p>1/64<br>
23/32<br>
5-1/2 days afterward<br>
2-1/2 times greater</p>
</blockquote>

<p>Display complex, built-up fractions by centering them vertically between two parts of a paragraph.</p>

<p>Place a zero to the left of the decimal in fractions less than 1.</p>

<blockquote>
<p>0.125<br>
0.006</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--funding-opportunities">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--funding-opportunities" id="funding-opportunities">
        
        funding opportunities
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p>As of Fiscal Year 2025, DOE funding opportunity announcements (FOAs) are retired in accordance with the OMB update to the Uniform Guidance: Title 2 of the Code of Federal Regulations. DOE is now issuing notices of funding opportunities (NOFOs) for the Infrastructure Investment and Jobs Act, Inflation Reduction Act, and base funding. (Internal users, see <a href="https://usdoe.sharepoint.com/sites/hub-bil/SitePages/BIL-&amp;-IRA-Templates-&amp;-Guidance.aspx?&amp;OR=Teams-HL&amp;CT=1729527134285&amp;clickparams=eyJBcHBOYW1lIjoiVGVhbXMtRGVza3RvcCIsIkFwcFZlcnNpb24iOiI0OS8yNDA5MTIyMTMxOCIsIkhhc0ZlZGVyYXRlZFVzZXIiOmZhbHNlfQ%3D%3D">FAIT Templates &amp; Guidance on sharepoint.com)</a>.</p><p>Notice of funding opportunity should not be capitalized unless it is part of an official name or in a title. Please use this term and acronym sparingly to adhere to plain language guidance (e.g., calling it simply “funding” or a “funding opportunity”).</p><p><em><strong>Correct</strong></em></p><blockquote><p><em>HEADLINE: Notice of Funding Opportunity: $3 Million Announced To Deploy Battery-Powered Buses in Remote Community Schools</em></p><p><em>Body text: The U.S. Department of Energy today announced a $3 million funding opportunity to deploy battery-powered buses in remote communities’ school systems. You should submit a proposal for funding if you are … This is one of several notices of funding opportunities (NOFOs) designed to ...</em></p></blockquote><p><em><strong>Incorrect</strong></em></p><blockquote><p>Body text: <em>The U.S. Department of Energy today announced a $3 million Notice of Funding Opportunity (NOFO) to deploy battery-powered buses in remote communities’ school systems. The NOFO seeks proposals from …</em></p></blockquote></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="g">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--g">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--g" id="g">
        
        G
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="G"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--geofluid">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--geofluid" id="geofluid">
        
        geofluid
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p>Any subsurface fluid, such as groundwater, geothermal fluids, basinal brines, petroleum, or magmatic fluid.</p>

<blockquote>
<p>This humidified cycle can make use of low-temperature geofluid water, normally applied only for heating, to generate power at higher geofluid efficiencies than typical geothermal cycles.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--geographic-information-system-gis">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--geographic-information-system-gis" id="gis">
        
        geographic information system, GIS
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="gis"></a>Do not capitalize "geographic information system" unless used as part of a proper noun. Do not use "geographical." GIS is considered a well-known acronym and can be used on first reference.</p>

<blockquote>
<p>She developed a GIS database, verified and validated by a third party, which displays annual and monthly average wave power.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--geographic-regions">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--geographic-regions" id="geographic">
        
        geographic regions
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="geographic"></a>Capitalize regions of the United States when they appear by themselves. See <a href="#states">states and countries</a>.</p>

<blockquote>
<p>the East, the West, the North, and the South<br>
the Southeast, the Northeast, the Southwest, the Pacific Northwest<br>
the Midwest, the East Coast, the West Coast, the Gulf Coast</p>
</blockquote>

<p>Don’t capitalize words that merely describe general areas in the country or areas of a state.</p>

<blockquote>
<p>the eastern United States<br>
southwestern Nebraska<br>
northern Atlanta<br>
the midwestern states</p>
</blockquote>

<p>Capitalize widely known geographic sections.</p>

<blockquote>
<p>Southern California<br>
West Texas<br>
the South Side of Chicago<br>
the Lower East Side of New York</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--geopressured-geothermal-resource">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--geopressured-geothermal-resource" id="geopressured">
        
        geopressured geothermal resource
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="geopressured"></a>In oil and gas production, co-produced fluid (brine) may be trapped under an impermeable layer of caprock while a layer of sediment rapidly builds over it. The weight of the sediment layer on the trapped fluids results in elevated pressures. These fluids are called geopressured resources; temperatures typically range from 90°C to 200+°C.</p>

<blockquote>
<p>An onsite geopressured-geothermal resource hybrid cycle 1 MW plant used a mixture of methane and geothermal brine fluids.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--geothermal-electric-technology-evaluation-model-getem">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--geothermal-electric-technology-evaluation-model-getem" id="getem">
        
        Geothermal Electric Technology Evaluation Model, GETEM
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="getem"></a>Built by Idaho National Laboratory, GETEM is an Excel-based tool used to estimate the Levelized Cost of Energy for definable geothermal scenarios.</p>

<blockquote>
<p>Electrical power generation is the sole geothermal use considered by GETEM, which does not provide assessment capabilities for geothermal direct-use or geothermal heat pumps.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--geothermal-heat-pump-ghp">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--geothermal-heat-pump-ghp" id="ghp">
        
        geothermal heat pump, GHP
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="ghp"></a>Geothermal heat pumps (GHPs), sometimes referred to as ground-source or water-source heat pumps, have been used as heating and cooling appliances since the late 1940s. They use the constant temperature of the earth as the exchange medium instead of the outside air temperature.</p>

<blockquote>
<p>The heating efficiency of geothermal heat pumps (GHP) is indicated by their coefficient of performance (COP), which is the ratio of heat provided in Btu per Btu of energy input.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--gigawatt-gw">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--gigawatt-gw" id="gigawatt">
        
        gigawatt, GW
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="gigawatt"></a>A unit of power equal to 1 billion watts, 1 million kilowatts, or 1,000 megawatts.</p>

<blockquote>
<p>In the first nine months of 2019, the United States installed 7.1 GW of photovoltaic power.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--glossaries-and-nomenclatures">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--glossaries-and-nomenclatures" id="glossaries">
        
        glossaries and nomenclatures
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="glossaries"></a>If a report or paper includes many mathematical or Greek symbols or technical terms, consider defining them in a glossary or nomenclature. Arrange the list alphabetically, and group Greek letters and definitions alphabetically in a separate list. Nomenclatures are usually in the front of a report, before the Table of Contents. Glossaries usually go in the back, before the references.</p>

<p>A less formal List of Terms may be used in place of the more formal glossary or nomenclature.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--google-earth-and-google-maps">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--google-earth-and-google-maps" id="google">
        
        Google Earth and Google Maps
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="google"></a>Google Earth satellite images and Google Maps terrain images need attribution, which is included in Google images with copyright notices such as “©2019 Google, Map Data ©2020 Tele Atlas.” Remove the Google logo and attribution text if already included in image content.</p><p>See <a href="https://www.google.com/permissions/geoguidelines/attr-guide/">Google Earth Permissions</a> for latest guidance.</p><p>In print, if attribution does not fit on the image or map, put separate attribution text directly adjacent to it.</p><p>In video, attribution must appear on-screen for the duration the map or image is shown; including attribution credits only does not suffice.</p><p>If the Google Earth image is altered (e.g., text or graphics added), the image is legal only if Google Earth software made the alteration and the correct attribution is included. Any other image alteration using other software is strictly prohibited.</p><p>Derivative works are prohibited. For example, CMEI cannot combine multiple static map images into one larger map.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--government">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--government" id="government">
        
        government
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="government"></a>Never capitalize (unless first word of a sentence) and never abbreviate. See also <a aria-label="https://www.energy.gov/eere/communicationstandards/style-guide-full-text#federal" href="#federal">federal</a>.</p>

<blockquote>
<p>the federal government<br>
a state government<br>
the U.S. government</p>
</blockquote></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="h">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--h">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--h" id="h">
        
        H
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="H"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--heat-mining">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--heat-mining" id="heat-mining">
        
        heat mining
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="heat-mining"></a>A concept (analogous) to convey harvesting or producing the planet’s natural geothermal resources (thermal energy).</p>

<blockquote>
<p>Accordingly, if current experimental techniques are perfected, the thermal energy could be "mined" from those vast areas of the Earth containing sufficiently hot rock but insufficient available water.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--hybrid-electric-vehicle-hev">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--hybrid-electric-vehicle-hev" id="hev">
        
        hybrid electric vehicle, HEV
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="hev"></a>This phrase contains no hyphens.</p>

<blockquote>
<p>Hybrid electric vehicles (HEVs) combine the benefits of gasoline engines and electric motors.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--hyphens">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--hyphens" id="hyphens">
        
        hyphens
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="hyphens"></a>No hyphen is needed between many prefixes and suffixes and the root words, unless the root word is a proper noun. <a href="#compound">See compound modifiers</a>. Use a hyphen between prefixes and proper nouns (but not common nouns) or dates, whether they’re used as nouns or modifiers.</p><blockquote><p>non-CMEI<br>mid-1990s<br>trans-American<br>pre-Civil War</p></blockquote><p><em>No hyphen</em></p><blockquote><p>nonprofit<br>prescreening<br>reuse</p></blockquote><p>Use a hyphen for clarity or to prevent confusion. Consider: "The scientists tested a new defect causing gas," versus "The scientists tested a new defect-causing gas."</p><p>In the first example, the scientists might seem to have been testing a defect; in the second example, it’s clear that they have tested a gas.</p><p>See <a href="#compound">compound words and modifiers</a>.</p><p>Use two hyphens when adding a prefix to a word that already contains a prefix, even when there is no hyphen after the prefix in the original word.</p><blockquote><p>non-self-limiting<br>sub-sub-head</p></blockquote><p>Don’t use a hyphen in common unit modifiers that are not ambiguous or confusing.</p><blockquote><p>high school students<br>solar radiation resource<br>solar thermal electric systems<br>renewable energy future<br>cold climate heat pump technology</p></blockquote><p>Don’t use a hyphen when multiple words of a compound modifier are capitalized.</p><blockquote><p>Bronze Age tools<br>Biofuels Program objectives<br>Nobel Peace Prize nomination</p></blockquote><p>Don’t use a hyphen with a compound modifier containing an adverb ending in “-ly.”</p><blockquote><p>highly efficient motor<br>frequently missed deadlines<br>heavily skewed results</p></blockquote><p>Omit hyphens if a sentence is rewritten so that the compound modifier comes after the noun it describes.</p><blockquote><p>We purchased state-of-the-art lab equipment.<br>We purchased lab equipment that reflects the state of the art.<br>They made some last-minute adjustments.<br>They made some adjustments at the last minute.</p></blockquote><p>When numbers are used in modifiers, retain all the necessary hyphens, or rewrite the sentence to omit the hyphens.</p><p><em>Numbers with modifiers and hyphens</em></p><blockquote><p>2-ft.-diameter pipes<br>13-cm-wide substrate</p></blockquote><p><em>Numbers with modifiers but without hyphens</em></p><blockquote><p>pipes that are 2 ft. in diameter<br>a substrate that is 13 cm wide</p></blockquote><p>Use a hyphen if the prefix ends in a vowel and the word that follows begins with the same vowel.</p><blockquote><p>co-op<br>intra-arterial<br>co-occur<br>anti-immune</p></blockquote><p><em>Exceptions</em></p><blockquote><p>cooperate, coordinate, and double-e combinations such as preestablish, preeminent, preeclampsia, preempt</p></blockquote></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="i">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--i">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--i" id="i">
        
        I
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="I"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--ie-and-eg">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--ie-and-eg" id="ie">
        
        i.e. and e.g.
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="ie"></a>See <a href="#eg">e.g. and i.e</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--insure-assure-and-ensure">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--insure-assure-and-ensure" id="insure">
        
        insure, assure, and ensure
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="insure"></a>See <a href="#assure">assure, ensure, and insure</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--intermittent">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--intermittent" id="intermittent">
        
        intermittent
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="intermittent"></a>Do not use "intermittent" when referring to wind energy. Instead, use the word "variable." "Intermittent" implies a binary on/off condition, whereas “variable” encompasses the way wind changes speed and direction</p>

<p><em>Correct</em></p>

<blockquote>
<p>The power grid was experiencing intermittent outages.</p>
</blockquote>

<p><em>Incorrect</em></p>

<blockquote>
<p>Intermittent wind caused insufficient power generation..</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--internet">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--internet" id="internet">
        
        internet
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="internet"></a>Lowercase "internet."</p>

<blockquote>
<p>The story was all over the internet.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--inverter">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--inverter" id="inverter">
        
        inverter
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="inverter"></a>A device that converts direct current (DC) electricity to alternating current (AC), either for stand-alone systems or to supply power to an electricity grid.</p>

<blockquote>
<p>DOE has funded R&amp;D for smart-grid ready inverters that capture the full value of distributed photovoltaic (PV).</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--italics">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--italics" id="italics">
        
        italics
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="italics"></a>Use italics (sparingly) to emphasize a word or phrase or bring attention to it.</p>

<blockquote>
<p><em>Never </em>operate equipment that has a yellow danger tag.</p>
</blockquote>

<p>Use italics for foreign words and phrases, such as <em>in situ</em>, <em>in vivo</em>, and <em>inter alia</em>; however, if the word or phrase is commonly used in your field, omit the italics.</p>

<p>Use italics for hyphenated prefixes to chemical formulas, such as <em>cis-</em>, <em>trans-</em>, <em>o-</em>, <em>m-</em>, and <em>p-</em>.</p>

<blockquote>
<p><em>trans</em>-1, 2-dibenzoylethylene<br>
<em>cis</em>-2-butene</p>
</blockquote>

<p>Italicize titles of long-form published or publicly released documents, including reports.</p>

<blockquote>
<p><em>2016 Renewable Energy Data Book<br>
Virginia Residential Energy Code Field Study: Baseline Report</em></p>
</blockquote>

<p>Use italics for book titles and the names of journals, newspapers, and magazines, in main text and in references, footnotes, and bibliographies. Titles of journal and magazine articles are given in regular type within quotation marks.</p>

<blockquote>
<p>Cohen, M.A., P.A. Kauzmann, and D.S. Callaway. 2016. "Effects of Distributed PV Generation on California’s Distribution System, Part 2: Economic Analysis." <em>Solar Energy</em> 128, 139–152, 2016.<br>
<em>Applied Physics Letters</em><br>
<em>The Denver Post</em><br>
"Solar Chimney Theory: Basic Precepts"</p>
</blockquote>

<p>Use italics in taxonomic names. Unless you’re discussing a genus in a general way, use italics to refer to specific genera, species, and varieties.</p>

<blockquote>
<p><em>Clostridium thermocellum<br>
C. thermocellum</em></p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--its-and-its">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--its-and-its" id="its">
        
        it&#039;s and its
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="its"></a>"It's"&nbsp;is a contraction of two words, "it is." "Its" is the possessive form of "it." Similar to "his," "hers," and "ours," the possessive "its" never includes an apostrophe.</p>

<blockquote>
<p>It's a shame that the company lost its biggest investor.</p>
</blockquote></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="k">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--k">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--k" id="k">
        
        K
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="K"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--kalina-cycle">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--kalina-cycle" id="kalina">
        
        Kalina cycle
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="kalina"></a>Kalina is always capitalized. Developed by Dr. Alexander Kalina, the Kalina cycle is a thermodynamic process for converting thermal energy into usable mechanical power.</p>

<blockquote>
<p>The Kalina cycle uses a solution of two fluids with different boiling points for its working fluid.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--kelvin-k">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--kelvin-k" id="kelvin">
        
        kelvin, K
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="kelvin"></a>Lighting color temperature is measured in kelvin (K) temperature. When referring to the Kelvin temperature scale, use a capital K. If referring to the unit of temperature, lowercase when spelling out but capitalize the symbol. Express kelvins as K with no degree sign; leave a space before the K.</p>

<blockquote>
<p>The kelvin is the fundamental unit of the Kelvin scale.</p>

<p>Higher kelvin temperatures (3600–5500 K) are what we consider cool and lower color temperatures (2700–3000 K) are considered warm.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--kilowatt-kw">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--kilowatt-kw" id="kilowatt">
        
        kilowatt, kW
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="kilowatt"></a>One thousand watts. As with all measurements, spell out on first mention.</p>

<blockquote>
<p>An electric heater consuming 1000 watts (1 kilowatt) and operating for one hour uses one kWh of energy.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--kilowatt-hour-kwh">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--kilowatt-hour-kwh" id="kilowatt-hour">
        
        kilowatt-hour, kWh
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="kilowatt-hour"></a>The kilowatt-hour is a unit of energy equal to one kilowatt of power sustained for one hour.</p>

<blockquote>
<p>Electrical energy is typically sold to consumers in kilowatt-hours (kWh).</p>
</blockquote></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="l">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--l">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--l" id="l">
        
        L
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="L"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--laboratory-and-lab">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--laboratory-and-lab" id="lab">
        
        laboratory and lab
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="lab"></a>CMEI follows DOE style, which capitalizes "laboratory" or "lab" when used with a laboratory’s proper name and in reference to the accredited U.S. National Labs. Lowercase in all other references. Exceptions may be made per Front Office preference. See <a href="#national-laboratory">National Laboratory</a>.</p><blockquote><p>The National Laboratory of the Rockies (NLR) is a U.S. Department of Energy National Laboratory. The lab is known for its research and development in energy technologies.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--life-cycle">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--life-cycle" id="life-cycle">
        
        life cycle
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="life-cycle"></a>Life cycle is two words and is never hyphenated, even in adjective form, per AP style.</p>

<blockquote>
<p>life cycle studies</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--light-bulb">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--light-bulb" id="light-bulb">
        
        light bulb
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="light-bulb"></a>Light bulb is two words.</p>

<blockquote>
<p>A-type lamps are considered the classic type of light bulb used for general-purpose lighting.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--lists">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--lists" id="lists">
        
        lists
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="lists"></a>Lists may be numbered for ordered (e.g., chronological) content or bulleted for all other content. Use numbered or lettered lists instead of bullets when it's necessary to refer to those items elsewhere in the text.</p><p>Listed items should be parallel in construction (begin all the items in the list with the same part of speech, such as a verb or a noun).</p><p>Capitalize the first word following each number and end each item with a period or other terminal punctuation (e.g., !, ?), except when the listed items consist of short noun phrases or single words. In such cases, punctuate only the final item.</p><p>Generally, CMEI prefers AP Style over Chicago Style on punctuating lists, except when AP Style end-punctuates each short phrase or single-word item. CMEI makes an exception to this, preferring that short phrases and single-word lists omit end punctuation, except for the final list. Do not use commas or semicolons at the ends of listed items.</p><blockquote><p>The top three new CMEI projects that incorporate waste-reduction concepts include:</p><ul><li>Plastics upcycling, with a goal to develop solutions for highly recycle plastics or biodegradable bio-based plastics.</li><li>REMADE Institute, focused on technologies to mitigate the technical and economic barriers that prevent greater material recycling, recovery, remanufacturing, and reuse.</li><li>Waste reduction pilot, launched in July 2019 to address the 2.7 billion tons of industrial solid waste and more than 260 million tons of municipal solid waste the U.S. generates annually, including plastic waste.</li></ul><p>The process is to:</p><ol><li>Submit your request.</li><li>Receive acknowledgement.</li><li>Start work!</li></ol></blockquote><p>An alternative is to list a few items or procedures in paragraph format and number them (1) one, (2) two, (3) three, etc.</p><p>See <a href="#bullets">bullets</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--lithology">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--lithology" id="lithology">
        
        lithology
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="lithology"></a>The study of rocks, particularly the character of a rock formation.</p>

<blockquote>
<p>Examples are predominantly granitic in composition, but other metamorphic and igneous lithologies may also be suitable.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--low-e">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--low-e" id="low-e">
        
        low-e
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="low-e"></a>Abbreviation for "low-emissivity." It refers to low-e storm windows, which have a low-e coating or glazing on them—a thin layer deposited directly on the surface of one or more panes of glass. The coating increases the window's energy efficiency by reflecting radiant heat. The usage is always lowercase, unless it begins a sentence.</p>

<blockquote>
<p>Coated with an ultra-thin, virtually invisible layer of metal, low-e windows reflect infrared heat back into the home.</p>
</blockquote></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="m">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--m">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--m" id="m">
        
        M
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="M"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--manufacturing-usa®">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--manufacturing-usa®" id="manufacturing-usa">
        
        Manufacturing USA®
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="manufacturing-usa"></a>Manufacturing USA® comprises 14 national manufacturing institutes. Each institute is a unique public-private partnership, jointly funded by government and private industry, focused on a different advanced manufacturing technology area.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--mathematical-symbols">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--mathematical-symbols" id="mathematical-symbols">
        
        mathematical symbols
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="mathematical-symbols"></a>Leave a space on either side of mathematical symbols used as operation signs (minus, plus, times). The slash (a/b) or division sign is an exception.</p>

<p>Leave a space between numerals and symbols of measurement such as cm and Å.</p>

<blockquote>
<p>T<sub>in</sub> - T<sub>amb</sub></p>

<p>ºC × 1.8</p>

<p>24 cm</p>

<p>1.058 Å</p>
</blockquote>

<p>Do not leave a space between symbols such as &gt;, &lt;, and the numeral unless they are operation signs in an equation.</p>

<p>Do not leave a space between numerals and the symbols for degrees, dollars (or cents), and percent).</p>

<blockquote>
<p>&lt;0.1</p>

<p>2/3</p>

<p>32º</p>

<p>$100</p>

<p>17%</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--measurement-units">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--measurement-units" id="measurement-units">
        
        measurement units
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="measurement-units"></a>See <a href="#units-of-measurement">units of measurement</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--megawatt-mw">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--megawatt-mw" id="megawatt">
        
        megawatt, MW
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="megawatt"></a>A unit of power equal to one million watts, especially as a measure of the output of a power station. Megawatt should be spelled out on first mention and abbreviated MW after that. If megawatts is used only once, then do not give the initialism (MW).</p>

<blockquote>
<p>The global sales of electrolyzers in 2017 were estimated to be 100 megawatts (MW) per year.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--megawatt-hour-mwh">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--megawatt-hour-mwh" id="megawatt-hour">
        
        megawatt-hour, MWh
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="megawatt-hour"></a>A megawatt-hour (MWh) is a unit of measure of electric energy. A MWh is 1,000 kilowatt-hours (kWh). Megawatt-hour should be spelled out on first mention and abbreviated MWh after that.</p><blockquote><p>Shepherds Flat is expected to generate 1,800,000 megawatt-hours (MWh) of energy each year.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--memoranda-memorandum">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--memoranda-memorandum" id="memoranda">
        
        memoranda, memorandum
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="memoranda"></a>The plural of "memorandum" can be either "memoranda" or "memorandums."</p>

<blockquote>
<p>The White House issued several memoranda on the topic.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--metric-conversions">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--metric-conversions" id="metric-conversions">
        
        metric conversions
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="metric-conversions"></a>For quick online conversions of English units of measurement to metric units, see the <a href="http://www.digitaldutch.com/unitconverter/">Digital Dutch Unit Converter</a> or the <a href="http://convert.french-property.co.uk/">Internet French Property Measuring Units Converter Table</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--metric-system">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--metric-system" id="metric-system">
        
        metric system
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="metric-system"></a>Shorthand for the SI (Systeme International d'Unites; International System of Units). See <a href="#si-metric-system">SI (Metric) System</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--microgrid">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--microgrid" id="microgrid">
        
        microgrid
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="microgrid"></a>A microgrid is a local energy grid with control capability, which means it can disconnect from the traditional grid and operate autonomously. Microgrids can also be used for demand response, time-of-use pricing, and other reasons. Microgrid is one word.</p>

<blockquote>
<p>A microgrid allows communities to be more energy independent and, in some cases, more environmentally friendly.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--microseismic-events">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--microseismic-events" id="microseismic-events">
        
        microseismic events
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="microseismic-events"></a>Microseismic is one word, no hyphen. In seismology, a microseism is defined as a faint earth tremor. Since about 2000, the term has become strongly associated with passive seismic monitoring of oil and gas production effects, borehole-related phenomena, and especially hydraulic fracture stimulation or fracking. When a microearthquake occurs as a result of industrial activity such as mining or hydraulic fracturing, it is referred to as a microseismic event.</p>

<blockquote>
<p>In many cases, the pore pressure required to shear favorably oriented joints can be very low, and vast numbers of microseismic events occur as the pressure migrates away from the well bore in a preferred direction associated with the direction of maximum principal stress.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--months-and-years">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--months-and-years" id="months">
        
        months and years
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="months"></a>Capitalize months. When a month has a specific date, abbreviate months with more than five letters, followed by a period.</p><blockquote><p>The workshop is scheduled for Jan. 9, 2020.</p><p>Register for the webinar on March 15, 2023.</p></blockquote><p>Spell out months when used alone or only with the year; omit commas when the month and year appear together.</p><blockquote><p>The event took place in October 2018.</p></blockquote><p>When a phrase refers to a month, day, and year, set off the year with commas.</p><blockquote><p>Feb. 14, 2013, was the target date.</p></blockquote><p>Abbreviate all months in tables; however, omit the period.</p><table><thead><tr><th>Jan</th><th>Feb</th><th>Mar</th><th>Apr</th><th>May</th><th>Jun</th><th>Jul</th><th>Aug</th><th>Sep</th><th>Oct</th><th>Nov</th><th>Dec</th></tr></thead><tbody><tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr></tbody></table><p>&nbsp;</p><p>Use lowercase s (no apostrophe) to show the plural of a decade expressed with numerals.</p><blockquote><p>the 1990s</p><p>the mid-1990s</p><p>the ‘90s</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--multijunction-solar-cell">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--multijunction-solar-cell" id="multijunction">
        
        multijunction solar cell
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="multijunction"></a>This term is preferred over "tandem solar cell." A multijunction (MJ) solar cell is a solar cell with multiple p–n junctions made of different semiconductor materials.</p>

<blockquote>
<p>The new CSP equipment uses high-intensity silicon vertical multijunction solar cells.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--multiplication-symbols">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--multiplication-symbols" id="multiplication">
        
        multiplication symbols
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="multiplication"></a>Be as consistent as possible in using multiplication symbols; as appropriate, choose one symbol (× or ·) or omit the symbol and use proximity or parentheses: ab, (ab) (cd), etc. When using symbols × or ·, spaces should be inserted before and after each.</p></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="n">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--n">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--n" id="n">
        
        N
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="N"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--nation">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--nation" id="nation">
        
        nation
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="nation"></a>Always lowercase "nation" when referring to the United States.</p>

<blockquote>
<p>Our nation is a leader in renewable energy markets.</p>

<p>Air quality improved across the nation during the COVID-19 pandemic.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--national-laboratory">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--national-laboratory" id="national-laboratory">
        
        National Laboratory
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="national-laboratory"></a>DOE has 17 National Laboratories. CMEI follows DOE style, which only capitalizes "laboratory" or "lab" when used with a laboratory's full proper name or referencing those specific labs. Lowercase all other references. Exceptions may be made per Front Office preference. See <a href="#lab">laboratory and lab</a>.</p><blockquote><p>One important reason for establishing America's National Laboratory system immediately after World War II was to provide a home for large-scale, costly scientific facilities that universities could not afford.</p><p>Oak Ridge National Laboratory unveiled Summit as the most powerful and smartest scientific supercomputer of any research lab around the world.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--nonattainment">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--nonattainment" id="nonattainment">
        
        nonattainment
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="nonattainment"></a>Spell as one word, no hyphen.</p>

<blockquote>
<p>Fuel cell systems are an ideal power source for industries where a high premium is placed on the reliability of electric power, in regions where low emission levels are required (such as urban and nonattainment areas), and in grid-constrained areas.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--noncondensable-gas">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--noncondensable-gas" id="noncondensable-gas">
        
        noncondensable gas
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="noncondensable-gas"></a>Any gas that does not condense (change from vapor to liquid) under pressure.</p>

<blockquote>
<p>The steam, along with its noncondensable gases, is routed to the power plant and used to produce electricity for the Big Island of Hawaii.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--numbers">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--numbers" id="numbers">
        
        numbers
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="numbers"></a><strong>Units of measurement</strong> and mathematical expressions. Use numerals with units of measurement and time. Also see <a href="#time">time</a>, <a href="#dates">dates</a>, <a href="#percent">percent</a>, and <a href="#addresses">addresses</a>.</p><blockquote><p>2-1/2 hours<br>4.5 months<br>36 cm<br>87 years<br>6 liters<br>25 kW</p></blockquote><p>Use numerals to imply arithmetical values or manipulation.</p><blockquote><p>a factor of 3<br>multiplied by 2<br>a ratio of 4:5<br>values of 1 and 48</p></blockquote><p>Express measurement errors as: 6 nm ± 0.2 nm. Leave a space between the number and the unit of measurement (0.2 nm) and put spaces around the operation sign; when the measurement error appears by itself, omit the space between the sign and the number.</p><blockquote><p>6 nm ± 0.2 nm<br>The measurement error is ±0.2 nm.</p></blockquote><p><strong>Time</strong>. With units of time, spell out numbers less than 10 consistently (applies to outreach communications products, not technical reports and papers).</p><blockquote><p>five-year plan<br>two-hour test<br>three-week turnaround<br>30 minutes<br>five seconds</p></blockquote><p><strong>Aligning numbers.</strong> Align numbers with a common measurement unit by decimals inside a table column. Put a zero before the decimal in numbers smaller than one.</p><p>If all the numbers in a column do not share the same measurement unit, center the numbers in the column and specify the measurement unit. See <a href="#data">data in tables</a>.</p><p><strong>Fractions and decimals. </strong>Spell and hyphenate simple fractions or express them (more complex fractions) in numerals with a slash.</p><blockquote><p>one-fifth or 1/5<br>1/64 (but not 1/64th)</p></blockquote><p>Use a hyphen to separate the integral and fractional parts of a mixed number or, convert the fraction to a decimal.</p><blockquote><p>2-1/2 cm in diameter<br>2.5-cm-diameter solar cell</p></blockquote><p>For numbers of 1 million or more, use the numeral (and a decimal, if necessary) and the words million, billion, etc.</p><blockquote><p>1.1 million households<br>3.5 billion people<br>$2.5 million in funding</p></blockquote><p><strong>Precision and numbers.</strong> Decimalization should not exceed two places in textual material unless there are special circumstances, e.g., measurement uncertainly analysis, which calls for measurement precision to a significant digit to the right of a decimal point, such as two or three digits (hundredths or thousandths). <strong>Check with an expert before changing the number of digits to the right of the decimal or rounding the numbers.</strong> For amounts less than 1, use the numeral zero before the decimal point. See also <a href="#standard-errors">standard errors</a>.</p><blockquote><p>Energy reduction per pole in this area was 59.3% compared to the QMH lighting system.<br>0.96</p></blockquote><p><strong>Punctuating numbers.</strong> Use a comma to separate groups of three digits in numbers.</p><blockquote><p>5,182<br>113,728<br>2,225,000</p></blockquote><p><strong>Ranges of numbers.</strong> Use an en dash (which is shorter than an em or long dash) with no spaces. If the word "of" or "from" occurs before the range, then use the word "to."</p><p>To express a range between some number and another number, always use "and" with "between."</p><p>Note that some symbols, e.g., º and %, are repeated in a range.</p><blockquote><p>15%–25%<br>6–12 cm<br>10–20 m<sup>2</sup><br>from 32º to 40ºC<br>from 66 to 80 V<br>between 8 and 12 m (not "between 8 to 12 m")<br>$3 million–$4 million</p></blockquote><p><strong>Scientific notation.</strong> Express multiples of SI (metric) units in powers of 10 with the prefix and technical abbreviations.</p><blockquote><p>mm (millimeters, 10<sup>-3</sup> m)<br>MJ (megajoules, 10<sup>6</sup> J)</p></blockquote><p><strong>Spelling out numbers.</strong> Except with units of measurement and time, spell out numbers less than 10. CMEI prefers "more than" to "over" to indicate greater numerical value. See <a href="#units-of-measurement">units of measurement</a>, <a href="#time">time</a>.</p><blockquote><p>eight experimental runs<br>three species of yeast<br>Salaries went up more than $20 a week. (preferred)<br>Salaries went up over $20 a week.</p></blockquote><p>Spell out all numbers at the beginning of a sentence.</p><blockquote><p>Fifteen trials later, the results were the same.</p><p>Thirty-five participants attended the seminar.</p></blockquote><p>When a sentence contains one or more numbers greater than nine that are related to a smaller number, use numerals for all of them.</p><blockquote><p>The results were the same in 3, 12, and 18 trials.<br>The contractor tested 8 devices in May, 12 in June, and 9 in July.</p></blockquote><p>Spell out the first of two adjacent numbers unless the first one requires three or more words. See also <a href="#fractions">fractions</a>.</p><blockquote><p>ten 5-kW arrays<br>thirty-two 4-cm<sup>2</sup> devices<br>135 16-cm collectors</p></blockquote></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="p">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--p">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--p" id="p">
        
        P
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="P"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--parentheses">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--parentheses" id="parentheses">
        
        parentheses
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="parentheses"></a>Use parentheses as appropriate for explanatory material in text.</p>

<p><strong>Parentheses in equations.</strong> In equations, use parentheses, brackets, and braces in this sequence (which may be repeated as needed).</p>

<blockquote>
<p>{[( )]}</p>
</blockquote>

<p><strong>Parentheses with measurements.</strong> Use parentheses around English measurements that follow SI (metric) measurements.</p>

<blockquote>
<p>3.1 m/s (7 mph)</p>
</blockquote>

<p><strong>Parentheses in citations.</strong> When using parentheses in text, such as for author-date references or for parenthetical (added) information, place a comma after the parentheses, not before them.</p>

<blockquote>
<p>In earlier research (Jones 1989), we showed how quantities of lipids could be increased by this method.</p>
</blockquote>

<p><strong>Nested parentheses in text.</strong> In body copy, use parentheses, brackets, and braces in this sequence, which may be repeated as needed: ([{ }]).</p>

<blockquote>
<p>(The data presented here [originally derived from Mason {1998}] should not be used for location-specific analyses.)</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--pascal-pa">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--pascal-pa" id="pascal">
        
        pascal, Pa
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="pascal"></a>The standard SI unit for pressure or stress (also, bar). The pascal is a measure of perpendicular force per unit area. It is equivalent to one newton per square meter. A megapascal equals 1,000,000 pascals.</p>

<blockquote>
<p>The small rocket engine produces 100,000 (105) Pa of pressure.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--percent--and-percentage">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--percent--and-percentage" id="percent">
        
        percent, %, and percentage
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="percent"></a>Use the symbol % with numerals; use “percent” when spelling out numbers at the beginning of a sentence. To determine whether "percent" or % is singular or plural, consider the noun following it. If the next noun is a plural, use a plural verb; if it’s singular, use a singular verb.</p>

<p>Do not confuse percent with percentage point. A change from 10% to 13% is a rise of 3 percentage points. This is not equal to a 3% change; rather, it’s a 30% increase.</p>

<blockquote>
<p>The maximum glucose yield was 60%.<br>
Six percent of the pipes were rusty.<br>
More than 10% of that amount was allocated to planning.</p>
</blockquote>

<p>When there is no number, use the word "percentage," unless people in the field use a different terminology, such as "percent difference."</p>

<blockquote>
<p>This table shows the percentages of government buildings having solar roofs, by state.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--phenomena-phenomenon">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--phenomena-phenomenon" id="phenomena">
        
        phenomena, phenomenon
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="phenomena"></a>"Phenomenon" is singular; "phenomena" is plural.</p>

<blockquote>
<p>Surface and interface phenomena often control the properties of materials and device structures used in energy-harvesting and storage applications.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--phone-numbers">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--phone-numbers" id="phone">
        
        phone numbers
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="phone"></a>Do not use parentheses around area codes in phone numbers. Use hyphens to separate the digits in phone numbers.</p>

<blockquote>
<p>303-275-3658<br>
1-800-555-5555<br>
011-44-20-7535-1515<br>
303-275-3658, ext. 2.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--photographs">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--photographs" id="photographs">
        
        photographs
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="photographs"></a>When using an image, credit the photographer or other source for legal purposes. The credit should include at minimum, Name, Organization, and Year (if available). The photo credit has no end punctuation.</p><p>Provide a caption in conformance with the style format of the content. If the photograph is intended for web posting, an alternate text caption describing the image should be provided and it should describe what is happening in the photo, not repeat the caption. See <a href="#captions">captions</a>.</p><blockquote><p>Photo courtesy of John Smith, National Laboratory of the Rockies, 2018<br>Photo: John Smith, NLR, 2018</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--photovoltaics-and-photovoltaic-pv">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--photovoltaics-and-photovoltaic-pv" id="photovoltaics">
        
        photovoltaics and photovoltaic, PV
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="photovoltaics"></a>"Photovoltaics" is a singular noun. "Photovoltaic" is an adjective. The acronym "PV" can be a noun or an adjective, but do not pluralize it. PV is one form of solar electricity, but it is not interchangeable with that term. Solar electricity can also be generated from CSP technologies.</p>

<blockquote>
<p>Projects focus on a wide variety of topics including photovoltaics, concentrating solar-thermal power, and power electronics.</p>

<p>One example of market-transforming innovations from other industries is solar photovoltaic technology.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--plain-language">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--plain-language" id="plain-language">
        
        plain language
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p>Refer to <a href="/plain-language" data-entity-type="node" data-entity-uuid="b93b5a06-b874-4f90-98ed-3d57449990bf" data-entity-substitution="canonical">the U.S. Department of Energy's commitment</a> and the U.S. federal government's <a href="https://www.plainlanguage.gov/guidelines/">plain language website</a> to learn more about the Plain Writing Act of 2010, which&nbsp;requires federal agencies to write “clear Government communication that the public can understand and use,” and find plain language guidance.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--policymaker-policymaking">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--policymaker-policymaking" id="policymaker">
        
        policymaker, policymaking
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="policymaker"></a>"Policymaker" and "policymaking" are both spelled as one word.</p>

<blockquote>
<p>The option helps policymakers to enable energy reduction goals for buildings and to increase transparency and consumer information around the home buying process.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--pressure">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--pressure" id="pressure">
        
        pressure
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="pressure"></a>Use the standard SI unit for pressure or stress, which is the <a href="#pascal">pascal (Pa)</a> or the bar. The bar is a metric unit of pressure, but not part of the International System of Units (SI). Bar is defined as exactly equal to 100,000 Pa (100 kPa).</p>

<p>Non-SI units include psi (pounds per square inch), millimeters of mercury, torr, and atmospheres, which are still in relatively widespread use.</p>

<blockquote>
<p>The small rocket engine produces 100,000 (105) Pa of pressure.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--principal-and-principle">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--principal-and-principle" id="principal">
        
        principal and principle
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="principal"></a>"Principal" means "chief," "main," or "first," such as the principal investigator in a research project or the principal of a high school. "Principle" refers to a belief, value, or rule.</p>

<blockquote>
<p><br>
"Guidance for Developing Principal Design Criteria for Non-Light-Water Reactors" helps developers align their concepts with relevant NRC regulations for nuclear power plants.</p>

<p>The Energy Literacy video series highlights the seven essential principles of energy to help engage students in energy.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--psi-pounds-per-square-inch">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--psi-pounds-per-square-inch" id="psi">
        
        psi, pounds per square inch
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><blockquote>
<p><a id="psi"></a>The product hydrogen exits the PSA at 300 psi and is compressed for storage in metal cylinder storage tanks (2,500 psi max pressures).</p>
</blockquote></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="q">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--q">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--q" id="q">
        
        Q
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="Q"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--quotation-marks">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--quotation-marks" id="quotation">
        
        quotation marks
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="quotation"></a>Use quotation marks for direct quotes and the titles of articles. Use “curly” or “fancy” quotation marks instead of inch marks ("). In Microsoft Word, go to Insert --&gt; Symbols --&gt; Left (or Right) Double Quotation Mark.</p>

<p>Place commas (and periods) inside quotation marks; place semicolons, question marks, dashes, and exclamation points outside quotation marks unless they’re part of the quotation.</p>

<p>Use single quotation marks to indicate a quotation within material already enclosed in double quotation marks.</p>

<blockquote>
<p>"Let’s meet again in six months," the chairman said, "to discuss our progress."<br>
She presented a paper, "Materials Research in Silvered Polymer Reflectors."<br>
"The results are in," he said.<br>
"Can you hear me?" she asked.<br>
Did he really say, "I don’t believe you"?<br>
“Explain what you mean by 'confidence,'" she said.</p>
</blockquote>

<p><strong>Block quotations.</strong> When quotations are longer than two or three lines, begin them on the next line and indent them on each side. No quotation marks are needed around block quotations. Use standard double quotation marks for quotes within block quotations.</p>

<p>The history of earthquake science began in the late 19th century, as described by Chris Jones, below.</p>

<blockquote>
<p>The modern concept of earthquake mechanisms began in the 1880s, when G.K. Gilbert, an American geologist, theorized that earthquakes were the result of displacement along geological faults (Gilbert 1890). In 1910, H.F. Reid suggested that earthquakes were the result of a phenomenon called "elastic rebound", based on observations of the great 1906 San Francisco earthquake (Reid 1910). This theory states that an earthquake is generated by a rupture or sudden displacement along a fault strained beyond its elastic strength.*</p>
</blockquote>

<p>* Chris Jones, History of the Science of Earthquakes, GPO, Washington, D.C., 1982.</p>

<p>&nbsp;</p>

<p>With in-text quotations, place reference numbers, superscripts, and author-date citations outside quotation marks. Place them after the final punctuation of the last sentence in a block quotation.</p>

<blockquote>
<p>"EIA forecasts renewable penetration to reach 38 percent and 17 GW of energy storage capacity will be added to the U.S. electric grid by 2050."<sup>1</sup></p>
</blockquote>

<p>See <a href="#footnotes">footnotes</a>, <a href="#references">references and citations</a>.</p></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="r">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--r">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--r" id="r">
        
        R
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="R"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--rankine-cycle">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--rankine-cycle" id="rankine">
        
        Rankine cycle
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="rankine"></a>A Rankine cycle is a thermodynamic cycle used in steam turbines to convert heat energy into work. Rankine is a proper noun; always capitalize.</p>

<blockquote>
<p>Concentrating solar power (CSP) plants often rely on the Rankine cycle.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--ratios">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--ratios" id="ratios">
        
        ratios
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="ratios"></a>CMEI prefers using a colon to indicate a ratio. However, some industries (such as the American automotive industry) use a slash to express a ratio. Note: AP Style doesn’t use the colon, instead "a ratio of 2-to-1," "a 2-1 ratio," "a 2-1 majority."</p><blockquote><p>We prepared a 3:1 dilution.</p><p>The engine is designed to have an optimum air/fuel ratio.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--references-and-citations">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--references-and-citations" id="references">
        
        references and citations
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="references"></a>References cited must be used to credit the original source of a claim, quote, statistic, or data. This allow others to place the work in the context of the published literature and lend credibility to published work.</p><p>A complete citation source is required [author, year, title, and hyperlink, at a minimum]. Always add a URL if one exists, followed by a period, to the end of the reference. If space is tight, you can shorten the number of authors to "et al." after three. Place works consulted (but not cited in the text) in a bibliography.&nbsp;</p><p>For all CMEI technical publications, consistently use in-text citations (with references at the end) or footnotes. CMEI prefers in-text author-date references over numbered footnotes or endnotes, following <a href="https://www.chicagomanualofstyle.org/tools_citationguide/citation-guide-2.html">Chicago Manual of Style formatting</a>. To prepare a manuscript for a publisher other than CMEI, follow that publisher’s preferred reference style.&nbsp;</p><p>All CMEI-funded reports are required to include an attribution to the sponsoring program at the program or subprogram level. This means that articles or reports funded by CMEI through a financial assistance award (such as a grant or cooperative agreement), including national labs or other third parties, must include the appropriate <a href="https://www.energy.gov/eere/communicationstandards/eere-branded-publications-developed-third-parties">Publication Acknowledgement Statement</a>.</p><p>See <a href="#footnotes">footnotes</a>.</p><p><em>In-Text Citations</em></p><blockquote><p>Several experts (Wendt et al. 2019; McLing et al. 2019) have given a detailed description of RTES reservoir configuration, heat source requirements, heat recovery power cycle configuration, and operating principles.<sup>1,2</sup></p></blockquote><p><em>Footnote Citations</em></p><blockquote><p>Several experts have given a detailed description of RTES reservoir configuration, heat source requirements, heat recovery power cycle configuration, and operating principles.<sup>1,2,3</sup></p></blockquote><p><em>Author-Date References</em></p><blockquote><p>Wendt, D., H. Huang, G. Zhu, P. Sharan, K. Kitz, S. Green, J. McLennan, J. McTigue, and G. Neupane. 2019. <em>Flexible Geothermal Power Generation utilizing Geologic Thermal Energy Storage: Seedling Project Final Report.</em> Idaho National Laboratory Technical Report. INL/EXT-19-53931. URL.</p><p>McLing T. L., D. Wendt, P. Dobson, C. Doughty, N. Spycher, D. Roberson, and J. McLaughlin. 2019. <em>Dynamic Earth Energy Storage: Terawatt-Year, Grid-Scale Energy Storage Using Planet Earth as a Thermal Battery (GeoTES): Seedling Project Final Report.</em> Idaho National Laboratory Technical Report. INL/EXT-19-53932. URL.</p></blockquote><p><em>Footnote References</em></p><blockquote><p>1 Smith, J., J. Doe, T. Welsh, et al. 2021. "Title Capitalized Regardless of Publication Capitalization." <em>Journal Name</em> <em>Volume(Issue): page–page. URL.</em></p><p>2 Author. 2024. Webpage or Article Title Capitalized. Publisher. Date Published or Accessed in CMEI style format. URL.</p><p>3 Author. 2022. Report Title in Italics and Capitalized. City, State/Country: Publisher Name. Publication number. URL.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--rheology">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--rheology" id="rheology">
        
        rheology
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="rheology"></a>The branch of physics that deals with the deformation and flow of matter, especially the non-Newtonian flow of liquids and the plastic flow of solids.</p>

<blockquote>
<p>Samples of Tank 22 with various solids content will be prepared for settling and rheology.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--roadmap-road-map">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--roadmap-road-map" id="roadmap">
        
        roadmap, road map
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p>Roadmap should be used as one word when referring to a plan or document laying out progress or steps toward a goal. Road map should be two words when referring to a tool used to navigate or travel via roadways, such as by vehicle.</p><blockquote><p>This roadmap will help the United States reach net-zero emissions by 2035.<br>You can identify your nearest electric vehicle charging station using a road map.</p></blockquote></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="s">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--s">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--s">
        
        S
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="S"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--sankey-diagram">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--sankey-diagram" id="sankey">
        
        Sankey diagram
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="sankey"></a>A standard model used in science and engineering to represent heat balance, energy flows, material flows, and life cycle assessment of products. Sankey is a proper noun; always capitalize.</p>

<blockquote>
<p>The onsite generation Sankey diagram shows offsite inputs of fuel and electricity that flow to conventional boilers, combined heat and power (CHP), and other steam- and/or electricity-generating systems.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--scientific-notation">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--scientific-notation" id="scientific-notation">
        
        scientific notation
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="scientific-notation"></a>Standard scientific notation represents a number as a factor multiplied by a power of 10; 3,560,000 is expressed as 3.56 × 10<sup>6</sup>. This is useful for very large and very small numbers, especially in non-SI units. Also use certain standard prefixes, many of which are listed to the right with their abbreviations.</p>

<p>Choose a prefix that permits the numerical value to fall between 0.1 and 1,000 (62 kW rather than 62,000 W).</p>

<table>
	<tbody>
		<tr>
			<td>10<sup>24</sup></td>
			<td>yotta</td>
			<td>Y</td>
		</tr>
		<tr>
			<td>10<sup>21</sup></td>
			<td>zetta</td>
			<td>Z</td>
		</tr>
		<tr>
			<td>10<sup>18</sup></td>
			<td>exa</td>
			<td>E</td>
		</tr>
		<tr>
			<td>10<sup>15</sup></td>
			<td>peta</td>
			<td>P</td>
		</tr>
		<tr>
			<td>10<sup>12</sup></td>
			<td>tera</td>
			<td>T</td>
		</tr>
		<tr>
			<td>10<sup>9</sup></td>
			<td>giga</td>
			<td>G</td>
		</tr>
		<tr>
			<td>10<sup>6</sup></td>
			<td>mega</td>
			<td>M</td>
		</tr>
		<tr>
			<td>10<sup>3</sup></td>
			<td>kilo</td>
			<td>k</td>
		</tr>
		<tr>
			<td>10<sup>2</sup></td>
			<td>hecto</td>
			<td>h</td>
		</tr>
		<tr>
			<td>10<sup>1</sup></td>
			<td>deka</td>
			<td>da</td>
		</tr>
		<tr>
			<td>10<sup>-1</sup></td>
			<td>deci</td>
			<td>d</td>
		</tr>
		<tr>
			<td>10<sup>-2</sup></td>
			<td>centi</td>
			<td>c</td>
		</tr>
		<tr>
			<td>10<sup>-3</sup></td>
			<td>milli</td>
			<td>m</td>
		</tr>
		<tr>
			<td>10<sup>-6</sup></td>
			<td>micro</td>
			<td>μ</td>
		</tr>
		<tr>
			<td>10<sup>-9</sup></td>
			<td>nano</td>
			<td>n</td>
		</tr>
		<tr>
			<td>10<sup>-12</sup></td>
			<td>pico</td>
			<td>p</td>
		</tr>
		<tr>
			<td>10<sup>-15</sup></td>
			<td>femto</td>
			<td>f</td>
		</tr>
		<tr>
			<td>10<sup>-18</sup></td>
			<td>atto</td>
			<td>a</td>
		</tr>
		<tr>
			<td>10<sup>-21</sup></td>
			<td>zepto</td>
			<td>z</td>
		</tr>
		<tr>
			<td>10<sup>-24</sup></td>
			<td>yocto</td>
			<td>y</td>
		</tr>
	</tbody>
</table>

<p>&nbsp;</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--semicolons">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--semicolons" id="semicolons">
        
        semicolons
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="semicolons"></a>Semicolons indicate a stronger or more important break in the flow of words than a comma.</p>

<p>Use semicolons in compound sentences (those with more than one subject and more than one verb) without conjunctions. When clauses in a sentence are closely related in meaning, a semicolon is an appropriate dividing punctuation mark. Note that the words "and," "but," "or," and "nor" do not follow semicolons.</p>

<blockquote>
<p>It was difficult to reproduce the experiment; the material Smith and Jones used was not widely available.<br>
Of the 13 samples, only one did not degrade; others deteriorated an average of 8%.</p>
</blockquote>

<p>Use semicolons with conjunctive adverbs, except "yet" and "so," which are usually preceded by commas in a complex sentence. Use a semicolon before such conjunctive adverbs as "then," "however," "thus," "therefore," "hence," "accordingly," "moreover," "nevertheless," "consequently," "besides," "indeed," and "subsequently"; place a comma after the adverb. Use a semicolon before "i.e." ("that is") and "e.g." ("for example") and a comma after them when a clause (with a subject and verb) follows them; use a comma when a phrase or list follows.</p>

<p><em>Before conjunctive adverbs</em></p>

<blockquote>
<p>We used the Schartz-Metterklume method in the experiment; however, the problems with this method are well known.<br>
Energy requirements are often expressed in quads, or quadrillion Btu; therefore, this report describes the number of quads supplied annually by each option.</p>
</blockquote>

<p><em>Exception: so</em></p>

<blockquote>
<p>The contractor’s representative was out, so I left a message.</p>
</blockquote>

<p><em>Before e.g.,</em></p>

<blockquote>
<p>Bring any two items; e.g., sleeping bags and tents are in short supply.</p>
</blockquote>

<p>Use semicolons in a series. When items in a series contain internal punctuation (e.g., commas) or are very long, separate them with semicolons. In those cases, a conjunction can follow the last semicolon.</p>

<p><em>Internal punctuation</em></p>

<blockquote>
<p>The contaminants in the sample were TCE, 150 ppb; toluene, 220 ppb; and benzene, 265 ppb.</p>

<p>Promising new technologies demonstrated at the exposition included advanced wind turbines; polycrystalline, thick-film, and thin-film solar cells; fast-growing energy crops; and fuel cells.</p>
</blockquote>

<p><em>Long series</em></p>

<blockquote>
<p>The vendor assured us that the replacement parts were on order; that the parts would be delivered as soon as they arrived; and that the delay in shipment was unavoidable.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--si-metric-system">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--si-metric-system" id="si-metric-system">
        
        SI (metric) system
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="si-metric-system"></a>CMEI follows national and scientific society policies by using the SI (Systeme International d’Unites; International System of Units) or the metric system to express technical measurements. English units (also known as Imperial) may follow metric ones or be used alone in special cases, when appropriate for a publication’s audience. See also the National Institute of Standards and Technology.</p><blockquote><p>He vowed to walk 62 miles (100 kilometers) in a week.<br>He vowed to walk 100 kilometers (62 miles) in a week.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--slash-solidus">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--slash-solidus" id="slash">
        
        slash (solidus)
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="slash"></a>The slash (also solidus, slant, shilling mark, or virgule) is a versatile symbol that has mathematical as well as textual functions. The slash is specific to language (preferred: slash, stroke, virgule, oblique) while the solidus is specific to the fraction symbol and Unicode (encoding method).</p>

<p>Use a slash in fractions.</p>

<p>Use a slash to express a quotient in text when you do not need to use a displayed equation.</p>

<p>Use a slash in superscript and subscript fractions.</p>

<blockquote>
<p>These structures yield photoluminescence lifetimes that are related to bulk lifetime by the expression, x<sup>1/2</sup>.</p>
</blockquote>

<p>In text, use a slash to indicate some junctions, interfaces, and components.</p>

<p>With abbreviated units of measurement, the slash stands for "per." But spell out "per" when units of measurement are spelled out.</p>

<p><em>Junctions/interfaces</em></p>

<blockquote>
<p>gas/liquid interface<br>
1-butyl acetate/acetic acid/water (3:1:1)</p>
</blockquote>

<p><em>Per as slash</em></p>

<blockquote>
<p>2 g/cm<sup>2</sup><br>
355 W/m<sup>2&nbsp;</sup></p>
</blockquote>

<p><em>Per spelled out</em></p>

<blockquote>
<p>several cubic meters per second<br>
a few cents per kilowatt-hour</p>
</blockquote>

<p>&nbsp;</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--smart-grid">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--smart-grid" id="smart-grid">
        
        smart grid
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="smart-grid"></a>Lowercase "smart grid" unless it begins the sentence.</p>

<blockquote>
<p>A video on the smart grid explains its impacts.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--solar-conversion-efficiency">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--solar-conversion-efficiency" id="solar-conversion">
        
        solar conversion efficiency
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="solar-conversion"></a>Define in outreach communications products as "the percentage of sunlight striking a solar cell that is converted into electricity." A definition is often unnecessary in technical publications.</p>

<blockquote>
<p>Due to the many recent advances in solar cell technology over the last few years, average solar conversion efficiency has increased from 15% to 20%.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--solar-electricity">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--solar-electricity" id="solar-electricity">
        
        solar electricity
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="solar-electricity"></a>This term can be used interchangeably with "photovoltaic power," "PV power," or "PV electricity."</p>

<blockquote>
<p>Even if you can’t directly purchase and install a solar system because you rent your home, you may still benefit from switching to solar electricity.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--sources">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--sources" id="sources">
        
        sources
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="sources"></a>Include the sources of all figures and tables originally published by others, especially those outside CMEI. If figures or tables come from a copyrighted publication, permission may be needed to reproduce them. Add the source at the end of a figure caption or in a note following a table. See <a href="#references">references and citations</a>, <a href="#captions">captions</a>, <a href="#photographs">photographs</a>.</p><blockquote><p>Source: Hansen, W.L.; Pearton, S.J.; Haller, E.E. (1984). Appl. Phys. Lett. 44:606.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--spaces">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--spaces" id="spaces">
        
        spaces
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="spaces"></a>Use only one space after a colon and between sentences.</p>

<p><em>Correct</em></p>

<blockquote>
<p>Use only one space between one sentence and the next. This is the correct way to format the next sentence.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--standard-errors">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--standard-errors" id="standard-errors">
        
        standard errors
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="standard-errors"></a>Express standard measurement errors as shown. A space is used before and after the symbol.</p>

<blockquote>
<p>6.0 nm ± 0.2 nm</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--state-implementation-plan">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--state-implementation-plan" id="state-implementation-plan">
        
        state implementation plan
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="state-implementation-plan"></a>Capitalize "state implementation plan" only when a state or organization name precedes it.</p>

<blockquote>
<p>The Connecticut Department of Environmental Protection incorporated emission reduction strategies into its U.S. Environmental Protection Agency State Implementation Plan for air quality.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--states-and-countries">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--states-and-countries" id="states">
        
        states and countries
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="states"></a><strong>States.</strong> In text, consistently spell out states’ names rather than using U.S. Postal Service abbreviations. A comma is needed between a city and a state name, and also after a state or country name, unless ending a sentence.</p><p>D.C. may be used for the District of Columbia in text, in both formal and informal publications. When indicating the location of a press release or news story, use only Washington.</p><p>When including addresses or state names in full addresses (containing streets and cities), contact lists, reference lists, and bibliographies, however, use the following postal abbreviations:</p><p>AL, AK, AZ, AR, CA, CO, CT, DC, DE, FL, GA, HI, ID, IL, IN, IA, KS, KY, LA, ME, MD, MA, MI, MN, MS, MO, MT, NE, NV, NH, NJ, NM, NY, NC, ND, OH, OK, OR, PA, RI, SC, SD, TN, TX, UT, VT, VA, WA, WV, WI, WY (PR, VI).</p><p>Note: CMEI takes exception to AP Style, which never abbreviates eight states: Alaska, Hawaii, Idaho, Iowa, Maine, Ohio, Texas, and Utah.</p><blockquote><p>California (rather than CA), Colorado (rather than CO), Wyoming (rather than WY)</p><p>He was traveling from Nashville, Tennessee, to Austin, Texas, en route to his home in Albuquerque, New Mexico.</p><p>She was born in Padua, Italy, in 1980.</p></blockquote><p>Always lowercase state or states, per typical common noun capitalization.</p><blockquote><p>The residential building stock database contains modeled data on housing type and housing energy expenditures for each U.S. census tract, city, county, and state.</p><p>She lives in New York state.</p><p>The apples come from Washington state.</p></blockquote><p><strong>Countries.</strong> Do not abbreviate the names of countries (including the United States) when they are used as nouns. Use U.S. as the adjective form. See <a href="#us" aria-label="https://www.energy.gov/eere/communicationstandards/style-guide-full-text#us">United States and U.S.</a></p><blockquote><p>the United States</p><p>U.S. DOE program</p><p>U.S. population</p></blockquote><p>Always lowercase country or countries, per typical common noun capitalization.</p><blockquote><p>The average pool pump energy bill is probably higher in Florida than in many other areas of the country because of the long swimming season.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--statistical-terms">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--statistical-terms" id="statistical-terms">
        
        statistical terms
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="statistical-terms"></a>When referring to statistical or graphical terms, use a hyphen but no italics. Also, do not use capital letters.</p>

<blockquote>
<p>p-value</p>

<p>t-test</p>

<p>y-axis</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--supercritical-fluid">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--supercritical-fluid" id="supercritical-fluid">
        
        supercritical fluid
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="supercritical-fluid"></a>A supercritical fluid is any substance at a temperature and pressure above its critical point, where distinct liquid and gas phases do not exist, but below the pressure required to compress it into a solid.</p>

<blockquote>
<p>We explore the fundamental connection between properties of a supercritical fluid and observed behavior of the flow by comparing simulations of a supercritical carbon dioxide round jet to canonical simulations using an ideal gas model.</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--systems-integrator">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--systems-integrator" id="systems-integrator">
        
        systems integrator
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="systems-integrator"></a>The correct term is "systems integrator," not "system integrator." A systems integrator is a person or company that specializes in bringing together component subsystems into a whole and ensuring that those subsystems function together, a practice known as systems integration. They often solve problems of automation.</p>

<blockquote>
<p>Systems integrators and installers should work to develop large megawatt-scale energy storage system (ESS) solutions for repurposed PEV batteries that minimize integration, balance of system, and installation costs.</p>
</blockquote></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="t">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--t">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--t" id="t">
        
        T
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="T"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--tandem-solar-cell">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--tandem-solar-cell" id="tandem-solar-cell">
        
        tandem solar cell
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="tandem-solar-cell"></a>The preferred term is "multijunction solar cell." See <a href="#multijunction">multijunction solar cell</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--taxonomic-names">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--taxonomic-names" id="taxonomic">
        
        taxonomic names
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="taxonomic"></a>See <a href="#capitalization">capitalization</a> and <a href="#italics">italics</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--temperature">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--temperature" id="temperature">
        
        temperature
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="temperature"></a>Use a degree symbol (º) with temperatures expressed in the Celsius and Fahrenheit scales but not with kelvins (use K). Don’t leave a space between the number and the letter for ºC and ºF, but leave a space between the number and K. See also <a href="#degree">degree symbol</a>.</p>

<blockquote>
<p>72°F</p>

<p>0 K</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--that-and-which">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--that-and-which" id="that">
        
        that and which
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="that"></a>See <a href="#which">which and that</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--iii-v-solar-cell">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--iii-v-solar-cell" id="iii-v">
        
        III-V solar cell
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="iii-v"></a>This term refers to a cell composed of semiconducting materials from Group III (e.g., gallium) and Group V (e.g., arsenic) elements of the periodic table.</p><blockquote><p>NLR develops technologies to drastically lower the cost of III-V solar cells while maintaining their conversion efficiency, thus enabling their use in conventional flat-plate and low-concentration applications.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--time-time-zone">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--time-time-zone" id="time">
        
        time, time zone
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="time"></a>Use lowercase a.m. and p.m. (with periods) to denote "ante meridiem" and "post meridiem" (before and after noon). Use a colon to separate hours from minutes except for the top of the hour. CMEI preferred style for time zone is ET, CT, MT, and PT (Eastern Time, Central Time, Mountain Time, and Pacific Time). Omit Daylight (D) and Standard (S) initials. Remove minutes if zero and write out "noon" and "midnight"; otherwise, use a colon before minutes.&nbsp;</p><blockquote><p>11 a.m. (not 11:00 a.m.)<br>3:30 p.m. PT</p></blockquote><p>Except when writing a news story, avoid time-sensitive phrases (such as "last week," or "this year"). If you feel the need to add time-sensitive phrases, use recent, soon, and current. Content that uses time-sensitive words rapidly becomes out-of-date. Specific dates are preferred.</p><p>See <a href="#numbers">numbers</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--title-professional">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--title-professional" id="title">
        
        title (professional)
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="title"></a>Titles (professional). Capitalize titles when they precede the person’s name. Lowercase titles and names of groups when they follow the name.</p><p>Former or future titles are also capitalized when preceding the person’s name. Except for "Acting," which CMEI prefers to capitalize, lowercase qualifiers (e.g., former, deposed, presumed).</p><blockquote><p>The Secretary of Energy was appointed in 2017.</p><p>Secretary of Energy John Doe attended the conference.</p><p>John Doe, the current secretary of energy, previously served as the deputy secretary of energy.</p><p>Chief Operating Officer Mark Wilson</p><p>Mark Wilson, the chief operating officer</p><p>former Secretary of State Henry Kissinger</p><p>Acting Chief of Police Robert Hand</p></blockquote><p>See <a href="#capitalization">capitalization</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--trademark-symbols">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--trademark-symbols" id="trademark">
        
        trademark symbols
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="trademark"></a>Use trademark symbols with trademarked programs and products owned/registered by DOE or affiliated national laboratories or lab operators. Use a trademark only on first mention; there is no space before the symbol.</p><p>Do not use trademark symbols (® or ™) with third-party products.</p><p>Commonly used government-owned trademarks that should use the trademark notice symbols are shown below. This list is not all-inclusive. For additional trademarked names and logos, refer to the CMEI branding team.</p><blockquote><p>BUILD4SCALE™</p><p>ENERGY STAR®</p><p>Energy Earthshots™</p><p>EnergyPlus® (word and design/logo registered trademarks)</p><p>H2@SCALE™</p><p>Home Energy Score™</p><p>HydroGEN Advancing Water Splitting Materials™</p><p>RACE TO ZERO™</p><p>RAPID TOOLKIT®</p><p>Solar Decathlon®</p><p>SOLSMART®</p></blockquote><p>&nbsp;</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--tribe-tribal-nation">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--tribe-tribal-nation" id="tribe">
        
        Tribe/Tribal Nation
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p>In keeping with the Office of Indian Energy, we recommend capitalizing "Tribe," "Tribal," "Native," "Nation," "Tribal Leader," "Tribal Council," and "Alaska Native Village," as well as "Alaska Native" as a matter of respect in all cases, whether part of a proper name or in indirect references to a specific Tribe (i.e., apart from the formal name).&nbsp;</p><blockquote><p>The Bishop Paiute Tribe is pursuing a solar project; the Tribal Nation applied for and was selected to receive technical assistance, and now, several other California Tribes are following suit. T</p><p><em>Representatives of several Villages attended the regional workshop on Alaska Native renewable energy development related to Native workers.</em></p></blockquote><p>"Tribal" should be capitalized when used as a proper noun or a generic modifier but not in references unrelated to Native American Tribes. "Nation" should be lowercase when referring to countries. You can lowercase "land" or "reservation" when referring to "Tribal land" or an "Indian reservation."</p><blockquote><p><strong>Do capitalize:</strong></p><p>The U.S. Department of Energy made new funding available to Tribal entities, including Tribally owned businesses.</p><p><strong>Do not capitalize:</strong></p><p>People rally around candidates along tribal lines across the nation.&nbsp;</p></blockquote></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="u">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--u">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--u" id="u">
        
        U
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="U"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--unit-modifiers">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--unit-modifiers" id="unit-modifiers">
        
        unit modifiers
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="unit-modifiers"></a>See <a href="#compound">compound words and modifiers</a>; see <a href="#hyphens">hyphens</a>.</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--united-states-and-us">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--united-states-and-us" id="us">
        
        United States and U.S.
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="us"></a>Spell out "United States" on first appearance as a noun, then use U.S. The abbreviation "U.S." is acceptable when it is used as an adjective, although "American" is preferred.</p><blockquote><p>The United States is a leader in advanced energy markets and supply chains are improving in the U.S.</p><p>The global energy markets are not as strong as the U.S. markets.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--units-of-measurement">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--units-of-measurement" id="units-of-measurements">
        
        units of measurement
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="units-of-measurements"></a>Use numerals with units of measurement and time in technical papers and reports, even when the number is less than 10. In some outreach communications products, spell out numbers less than 10, especially with units of time.</p>

<p>Except with $, °, and %, leave a space between the numeral and the unit.</p>

<p>Use non-SI (Systeme International d’Unites) or nonmetric measurement units (English or Imperial units) instead of metric units only when they are the industry standard. Otherwise, state metric units first, followed by English equivalents in parentheses.</p>

<p>For numbers of 1 million or more, use the numeral (and a decimal, if necessary) and the words million, billion, etc.</p>

<p>See <a href="#numbers">numbers</a>.</p>

<blockquote>
<p>2 kW</p>

<p>16.8%</p>

<p>3 m</p>

<p>8-hour days</p>

<p>300 Btu</p>

<p>7 cm<sup>2</sup></p>

<p>5 years</p>

<p>$2 billion</p>

<p>45°</p>

<p>38.1 m (125 ft)</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--url-uniform-resource-locator">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--url-uniform-resource-locator" id="url">
        
        URL, uniform resource locator
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="url"></a>Uniform resource locators, or URLs, are essentially web addresses. On websites, URLs should be embedded in text. "URL" is widely known and can be used without spelling out on first reference; however, CMEI preference is to use "website" over "URL" in copy.</p><p>In print, URLs should not be embedded in text. If a URL extends beyond one line of text, add a break at a slash. Also, in general, don’t include the https:// prefix on most URLs — but test before removing it. Shorten URLs as much as possible (e.g., remove unnecessary trailing such as /index.html) while ensuring functionality.</p><p>When embedding URLs as hyperlinks in digital content, best practice is to keep the hyperlink as short as possible, ideally three words or fewer.</p><blockquote><p>More information is available on the <a href="https://www.energy.gov/eere/wind/wind-energy-technologies-office">Wind Energy Technologies website</a>.</p><p>See <a href="https://www.energy.gov/">energy.gov</a> for more information.</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--us-department-of-energy-doe">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--us-department-of-energy-doe" id="usdoe">
        
        U.S. Department of Energy, DOE
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="usdoe"></a>The preferred name of the department for all outreach materials. Do not include "The" or "U.S." before the acronym "DOE."</p><p>On second reference, "the Department" may be used in place of the DOE initialism. Avoid using "the Department" if the content references other federal agencies that could be confused with the Department of Energy.</p><p>If the possessive is used with the name, the apostrophe should go after the spelled-out name or term as well as with the DOE initialism. Preference is to write in a way that avoids use of the possessive.</p><blockquote><p>The Department of Energy’s (DOE's) Office of Critical Minerals and Energy Innovation (CMEI) is in charge of the program.</p><p>Under the U.S. Department of Energy (DOE), the Office of Critical Minerals and Energy Innovation (CMEI) is in charge of the program.</p></blockquote><p>On first reference, and only on first reference, "Energy Department" may be used in communications to the press, including in news article headlines.</p></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="w">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--w">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--w" id="w">
        
        W
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="W"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--watt">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--watt" id="watt">
        
        watt
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p>Watt is the SI unit of power, equivalent to one joule per second, corresponding to the power in an electric circuit in which the potential difference is one volt and the current one ampere. The symbol is W. See also <a href="#kilowatt">kilowatt</a>, <a href="#megawatt">megawatt</a>, <a href="#gigawatt">gigawatt</a>.</p>

<blockquote>
<p>1,000 watts of electricity</p>

<p>1,000-watt output</p>

<p>Luminous efficacy of solid-state lighting is measured in lumens per watt (lm/W).</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--web-terms">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--web-terms" id="web">
        
        web terms
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a class="ck-anchor" id="web"></a>Web terms (single and compound) are one word, lowercase.</p><p>World Wide Web is a proper noun and should be initial-capped. "The web" alone, if used as shorthand for the World Wide Web, does not need to be capitalized. It should also not be capitalized when used as an adjective, e.g., web-based training, web tool, web content.</p><blockquote><p>the web</p><p>webpage</p><p>homepage</p><p>webcast</p><p>webinar</p><p>webmaster</p><p>website</p></blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--which-and-that">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--which-and-that" id="which">
        
        which and that
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="which"></a>Standard American English uses "which" for nonrestrictive (nonessential) phrases and clauses and "that" for restrictive (essential) phrases and clauses.</p>

<p>When a phrase or clause is not essential to the meaning of a sentence, use the relative pronoun "which" and enclose the phrase or clause in commas.</p>

<p>When a phrase or clause is essential to the meaning of a sentence (that is, the sentence would not make sense without it), use "that" and leave out the commas.</p>

<p><em>Which</em></p>

<blockquote>
<p>The paper, which she has been working on for three weeks, discusses string theory.</p>
</blockquote>

<p><em>That</em></p>

<blockquote>
<p>The paper that he completed recently will be presented in New York; the paper that he finished last summer will be presented in Philadelphia.</p>
</blockquote>

<p>&nbsp;</p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--who-and-whom">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--who-and-whom" id="who">
        
        who and whom
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="who"></a>Who is the pronoun used for references to human beings and to animals with a name. Who is grammatically the subject (never the object) of a sentence, clause, or phrase.</p>

<p>Whom is used when someone is the object of a verb or preposition.</p>

<p><em>Who</em></p>

<blockquote>
<p>The woman who rented the room left the window open.</p>

<p>Who is there?</p>
</blockquote>

<p><em>Whom</em></p>

<blockquote>
<p>The woman, with whom she visited, left in a hurry.</p>

<p>Whom do you wish to see?</p>
</blockquote></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--work-for-others-agreement-wfo-agreement">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--work-for-others-agreement-wfo-agreement" id="wfo">
        
        work-for-others agreement, WFO agreement
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="wfo"></a>Lowercase "work-for-others agreement" because it’s not a proper noun. The acronym "WFO" refers only to "work for others;" therefore, "WFO agreement" is correct.</p>

<blockquote>
<p>Work-for-others (WFO) agreements permit DOE laboratories and facilities to conduct work for other federal agencies and non-federal entities (including state and local governments, universities) on a reimbursable basis.</p>
</blockquote></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="y">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--y">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--y" id="y">
        
        Y
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="Y"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--years-and-months">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--years-and-months" id="years">
        
        years and months
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="years"></a>See <a href="#months">months and years</a>.</p></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>








  

<section class="layout blade-section--boxed"  id="z">

    <div class="grid-container">
      <div class="blade-color blade-section--white" style="">
                                      <div  class="grid-row">
                  <div  class="grid-col">
      
        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--z">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h2 class="block block-layout-builder block-inline-blockbasic block-title--z" id="z">
        
        Z
                  </h2></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="Z"></a></p></div>
      
            </div>

        
  
  
<div class="block block-layout-builder block-inline-blockbasic block-title--zero">
        
  <div class="paragraph-reference__heading-subheading-wrapper"><div class="paragraph-reference-heading-wrapper paragraph-reference-heading-wrapper--"><h3 class="block block-layout-builder block-inline-blockbasic block-title--zero" id="zero">
        
        zero
                  </h3></div></div>  
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a id="zero"></a>For numbers less than one, place a zero before the decimal.</p>

<blockquote>
<p>0.5</p>

<p>0.125</p>

<p>0.00125</p>
</blockquote></div>
      
            </div>


<div class="block block-layout-builder block-inline-blockbasic">
        
    
          

            <div class="field field--text_default field--body field field--name-body field--type-text-with-summary field--label-hidden field__item"><p><a href="#top">Back to Top</a>&nbsp;|&nbsp;<a data-entity-substitution="canonical" data-entity-type="node" data-entity-uuid="60ce5a01-b3af-4c13-90b9-f36b8a2864e3" href="/cmei/communicationstandards/cmei-style-guide" title="CMEI Style Guide" aria-label="CMEI Style Guide">Topic Index</a></p></div>
      
            </div>

    </div>
  
        </div>
            </div>
    </div>
  </section>





  </div>

</article>

              
      </div>


                      </div>        
              </div>
    </div>
  </main>
  
<footer class="usa-footer usa-footer--medium" role="contentinfo">
      <div class="usa-footer__primary-section">
      <div class="usa-footer__primary-content grid-container">
        <div class="logo-tagline">
          <img src="/themes/custom/energy_gov/svg/footer-logo.svg" alt="Department of Energy" />
                      <p class="text-bold">
              Committed to Restoring America’s Energy Dominance.
            </p>
                  </div>
        <div class="footer-address">
          
        </div>
        <div class="footer-social-bar">
          <p class="footer-heading">Follow Us</p>
                      <a href="https://www.facebook.com/energy">
              <i class="fa-brands fa-facebook"></i>
              <span class="usa-sr-only">Link to Facebook</span>
            </a>
                                      <a href="https://www.instagram.com/energy">
                <i class="fa-brands fa-instagram"></i>
                <span class="usa-sr-only">Link to Instagram</span>
              </a>
                                      <a href="https://www.linkedin.com/company/u-s--department-of-energy">
                <i class="fa-brands fa-linkedin"></i>
                <span class="usa-sr-only">Link to Linkedin</span>
              </a>
                                      <a href="https://www.youtube.com/user/USdepartmentofenergy">
                <i class="fa-brands fa-youtube"></i>
                <span class="usa-sr-only">Link to Youtube</span>
              </a>
                                      <a href="https://twitter.com/energy">
                <i class="fa-brands fa-x-twitter"></i>
                <span class="usa-sr-only">Link to X</span>
              </a>
                                                  <a href="https://www.flickr.com/photos/departmentofenergy/">
                <i class="fa-brands fa-flickr"></i>
                <span class="usa-sr-only">Link to Flickr</span>
              </a>
                    </div>
        <div class="footer-newsletter">

        
        </div>
        <div class="footer-agency">
          <nav class="usa-footer__nav" aria-label="Our Agency">
            
      <ul class="menu menu--our-agency nav">
                              <li class="expanded dropdown first last">
                                                                                    <span class="expanded dropdown first last dropdown-toggle" data-toggle="dropdown">Quick Links <span class="caret"></span></span>
                            <ul class="dropdown-menu">
                              <li class="first">
                                                  <a href="/our-leadership-offices" class="first" data-drupal-link-system-path="node/4847008">Leadership &amp; Offices</a>
                  </li>
                              <li>
                                                  <a href="/newsroom" data-drupal-link-system-path="node/4847098">Newsroom</a>
                  </li>
                              <li>
                                                  <a href="/contact-us" data-drupal-link-system-path="node/268177">Contact Us</a>
                  </li>
                              <li class="last">
                                                  <a href="/work-us-department-energy" class="last" data-drupal-link-system-path="node/4847097">Careers</a>
                  </li>
          </ul>
  
                  </li>
          </ul>
  

          </nav>
        </div>
        <div class="footer-resources">
          <nav class="usa-footer__nav" aria-label="Our Resources">
            
      <ul class="menu menu--our-resources nav">
                              <li class="expanded dropdown first last">
                                                                                    <span class="expanded dropdown first last dropdown-toggle" data-toggle="dropdown">Resources <span class="caret"></span></span>
                            <ul class="dropdown-menu">
                              <li class="first">
                                                  <a href="/budget-performance" class="first" data-drupal-link-system-path="node/268189">Budget &amp; Performance </a>
                  </li>
                              <li>
                                                  <a href="https://www.directives.doe.gov/">Directives, Delegations, &amp; Requirements </a>
                  </li>
                              <li>
                                                  <a href="/gc/freedom-information-act" data-drupal-link-system-path="node/3049">Freedom of Information Act (FOIA)</a>
                  </li>
                              <li>
                                                  <a href="/ig/office-inspector-general" data-drupal-link-system-path="node/4847015">Inspector General </a>
                  </li>
                              <li class="last">
                                                  <a href="/cio/privacy-program" class="last" data-drupal-link-system-path="node/2603217">Privacy Program</a>
                  </li>
          </ul>
  
                  </li>
          </ul>
  

          </nav>
        </div>
        <div class="footer-policies">
          <nav class="usa-footer__nav" aria-label="Federal Policies and Information">
            
      <ul class="menu menu--federal-policies nav">
                              <li class="expanded dropdown first last">
                                                                                    <span class="expanded dropdown first last dropdown-toggle" data-toggle="dropdown">Federal Government <span class="caret"></span></span>
                            <ul class="dropdown-menu">
                              <li class="first">
                                                  <a href="https://www.usa.gov/" class="first">USA.gov</a>
                  </li>
                              <li class="last">
                                                  <a href="https://www.whitehouse.gov/" class="last">The White House </a>
                  </li>
          </ul>
  
                  </li>
          </ul>
  

          </nav>
        </div>
      </div>
    </div>
        <div class="usa-footer__secondary-section">
      <div class="usa-footer__secondary-content grid-container">
        <nav class="usa-footer__nav" aria-label="Utility Links">
          
      <ul class="menu menu--menu-footer-utility nav">
                              <li class="first">
                                                  <a href="/open-government" class="first" data-drupal-link-system-path="node/268195">Open Gov</a>
                  </li>
                              <li>
                                                  <a href="/cio/accessibility-standard-statement" data-drupal-link-system-path="node/723576">Accessibility</a>
                  </li>
                              <li>
                                                  <a href="/about-us/web-policies/privacy" data-drupal-link-system-path="node/268159">Privacy</a>
                  </li>
                              <li>
                                                  <a href="/cio/department-energy-information-quality-guidelines" data-drupal-link-system-path="node/289213">Information Quality</a>
                  </li>
                              <li>
                                                  <a href="/OMEI/no-fear-act-data" data-drupal-link-system-path="node/2713">No Fear Act</a>
                  </li>
                              <li>
                                                  <a href="/web-policies" data-drupal-link-system-path="node/268153">Web Policies</a>
                  </li>
                              <li>
                                                  <a href="/cio/articles/vulnerability-disclosure-policy" data-drupal-link-system-path="node/4802628">Vulnerability Disclosure Program</a>
                  </li>
                              <li>
                                                  <a href="/whistleblower-protection-and-nondisclosure-agreements" data-drupal-link-system-path="node/631441">Whistleblower Protection</a>
                  </li>
                              <li>
                                                  <a href="/OMEI/notice-equal-employment-opportunity-eeo-findings-discrimination-harassment-andor-retaliation" data-drupal-link-system-path="node/4823758">Equal Employment Opportunity</a>
                  </li>
                              <li class="last">
                                                  <a href="/notice-court-orders" class="last" data-drupal-link-system-path="node/4849328">Notice of Court Orders</a>
                  </li>
          </ul>
  

        </nav>
      </div>
    </div>
  </footer>

  </div>

    <script>window.dataLayer = window.dataLayer || []; window.dataLayer.push({"drupalLanguage":"en","drupalCountry":null,"siteName":"Department of Energy","entityCreated":"1442424686","entityLangcode":"en","entityStatus":"1","entityUid":"130951","entityUuid":"4e34d733-ffab-40c1-ba42-9bb0e977adcb","entityVid":"1756407","entityName":"charles.joyner@doe.gov","entityType":"node","entityBundle":"page","entityId":"1246196","entityTitle":"Style Guide Full Text","userUid":0,"primaryOffice":{"id":"867621","label":"Communication Standards and Guidelines"}});</script>

    <script type="application/json" data-drupal-selector="drupal-settings-json">{"path":{"baseUrl":"\/","pathPrefix":"","currentPath":"node\/1246196","currentPathIsAdmin":false,"isFront":false,"currentLanguage":"en"},"pluralDelimiter":"\u0003","suppressDeprecationErrors":true,"dataLayer":{"defaultLang":"en","languages":{"en":{"id":"en","name":"English","direction":"ltr","weight":0},"es":{"id":"es","name":"Spanish","direction":"ltr","weight":1}}},"external_link_popup":{"whitelist":"","width":"85%","popups":[{"id":"default","name":"Default","status":true,"weight":10,"close":true,"title":"You Are Now Leaving This Site","body":"\u003Cp\u003EWould you like to leave this site?\u003C\/p\u003E\n","labelyes":"OK","labelno":"Cancel","domains":"*","target":"_self"}]},"google_analytics":{"account":"G-GS1KB7R8DY","trackOutbound":true,"trackMailto":true,"trackDownload":true,"trackDownloadExtensions":"7z|aac|arc|arj|asf|asx|avi|bin|csv|doc(x|m)?|dot(x|m)?|exe|flv|gif|gz|gzip|hqx|jar|jpe?g|js|mp(2|3|4|e?g)|mov(ie)?|msi|msp|pdf|phps|png|ppt(x|m)?|pot(x|m)?|pps(x|m)?|ppam|sld(x|m)?|thmx|qtm?|ra(m|r)?|sea|sit|tar|tgz|torrent|txt|wav|wma|wmv|wpd|xls(x|m|b)?|xlt(x|m)|xlam|xml|z|zip"},"data":{"extlink":{"extTarget":true,"extTargetAppendNewWindowDisplay":false,"extTargetAppendNewWindowLabel":"(opens in a new window)","extTargetNoOverride":false,"extNofollow":true,"extTitleNoOverride":false,"extNoreferrer":true,"extFollowNoOverride":false,"extClass":"ext","extLabel":"(link is external)","extImgClass":false,"extSubdomains":true,"extExclude":"\\.gov","extInclude":"isExternalResource","extCssExclude":"footer, .social-bar-utility, .media-type-image, .energy-social-links","extCssInclude":"","extCssExplicit":"","extAlert":true,"extAlertText":"This link will take you to an external web site. We are not responsible for their content.","extHideIcons":false,"mailtoClass":"0","telClass":"","mailtoLabel":"(link sends email)","telLabel":"(link is a phone number)","extUseFontAwesome":false,"extIconPlacement":"append","extPreventOrphan":false,"extFaLinkClasses":"fa fa-external-link","extFaMailtoClasses":"fa fa-envelope-o","extAdditionalLinkClasses":"","extAdditionalMailtoClasses":"","extAdditionalTelClasses":"","extFaTelClasses":"fa fa-phone","allowedDomains":null,"extExcludeNoreferrer":""}},"energyAnalytics":{"accounts":["UA-17192165-1"],"pageTitle":"Style Guide Full Text","settings":[["_setAccount","UA-17192165-1"],["_trackPageview"],["_setDomainName",".energy.gov"]]},"searchSettings":{"bestBetsHeadingText":"Recommended by DOE:","defaultThumbnailUrl":"https:\/\/www.energy.gov\/sites\/default\/files\/2025-04\/white-fallback_0.png","enableTopicsAutocomplete":true,"events":{"showUpcomingEventsSection":true,"viewAllEventsLinkTitle":"View All Events","viewEventDetailsLinkTitle":"View Event Details"},"isSearchEventsPage":false,"isSearchEventsCalendarPage":false,"searchHeaderMessage":{"generic":"What are you looking for?","events":"What event are you looking for?"},"searchPlaceholderText":"Try \u0022Geothermal\u0022","popularSearchTerm1":"Artificial Intelligence","popularSearchTerm2":"LNG","popularSearchTerm3":"Geothermal","popularSearchTerm4":"Nuclear Energy","popularSearchTerm5":"Hydropower","popularSearchTerm6":"","facetsFooterMessage":"\u003Cp\u003E\u003Cstrong\u003ELooking for compliance docs?\u003C\/strong\u003E\u003Cbr\u003E\u003Ca href=\u0022\/compliance-documents\u0022\u003EVisit our Compliance Document search page\u003C\/a\u003E\u003C\/p\u003E","noResultsFoundMessage":"No matches were found using the filters you have selected.","suggestedTopicsMessage":"Suggested Topics based on Your Search","spellcheck":{"labelPrefix":"Did you mean: ","labelSuffix":"? Suggestion based on closest match(es)."},"timeZone":"America\/New_York","topics":{"topicsCountForSearchPage":"3"}},"sitewideAlert":{"refreshInterval":150000,"automaticRefresh":true,"showCount":false,"serverSideRender":false},"user":{"uid":0,"permissionsHash":"a4e0a35ce7684b916e6d99677b1277cc37e5d6c5662a1fdbb585b5a6e5e5688b"}}</script>
<script src="/sites/default/files/js/js_0zk0oGI9ENdXSAdeJvx-5Ug56SXWmgK8xWEiP0lAb58.js?scope=footer&amp;delta=0&amp;language=en&amp;theme=energy_gov&amp;include=eJxtjUEOwjAMBD9U4Qv_iUxipVYtJ7JNIL-nhwYuHHd3RlswUHCSwYN2HNzMN1KyOhMqygzODhXviQZppDDMB2v9w7RWhX7FInLTOE24Ir1DWI-11jYA3Sm-vBNa3gF73y4Wij07ym2pzkEvLueXkAWwcnwAqu9OzQ"></script>
<script src="/modules/custom/energy_search/js/search-uswds/build/static/js/bundle.js?tetb3h"></script>
<script src="/sites/default/files/js/js_nUBP9691_qoXGF4VbycV5zJz7stvGI1lXrkXm53x2-w.js?scope=footer&amp;delta=2&amp;language=en&amp;theme=energy_gov&amp;include=eJxtjUEOwjAMBD9U4Qv_iUxipVYtJ7JNIL-nhwYuHHd3RlswUHCSwYN2HNzMN1KyOhMqygzODhXviQZppDDMB2v9w7RWhX7FInLTOE24Ir1DWI-11jYA3Sm-vBNa3gF73y4Wij07ym2pzkEvLueXkAWwcnwAqu9OzQ"></script>

      </body>

      </html>
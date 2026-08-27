<!-- URL: https://news.usc.edu/ -->
<!-- HTTP: 200 SIZE: 209249 -->

<!doctype html>
<!--[if lt IE 9]><html class="no-js no-svg ie lt-ie9 lt-ie8 lt-ie7" lang="en-US"> <![endif]-->
<!--[if IE 9]><html class="no-js no-svg ie ie9 lt-ie9 lt-ie8" lang="en-US"> <![endif]-->
<!--[if gt IE 9]><!--><html class="no-js no-svg" lang="en-US"> <!--<![endif]-->
<head>
    <meta charset="UTF-8" />
    <meta name="description" content="University of Southern California News">
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge"><script type="text/javascript">(window.NREUM||(NREUM={})).init={privacy:{cookies_enabled:true},ajax:{deny_list:["bam.nr-data.net"]},feature_flags:["soft_nav"],distributed_tracing:{enabled:true}};(window.NREUM||(NREUM={})).loader_config={agentID:"1134378470",accountID:"4166130",trustKey:"66686",xpid:"UAcBV1dQCBABVVNaBAcAXlYD",licenseKey:"NRJS-37496976d3a7f44e435",applicationID:"1128367822",browserID:"1134378470"};;/*! For license information please see nr-loader-spa-1.314.0.min.js.LICENSE.txt */
(()=>{var e,t,r={384:(e,t,r)=>{"use strict";r.d(t,{NT:()=>a,Zm:()=>c,bQ:()=>u,dV:()=>d,pV:()=>l});var n=r(6154),i=r(1863),s=r(944),o=r(1910);const a={beacon:"bam.nr-data.net",errorBeacon:"bam.nr-data.net"};function c(){return n.gm.NREUM||(n.gm.NREUM={}),void 0===n.gm.newrelic&&(n.gm.newrelic=n.gm.NREUM),n.gm.NREUM}function d(){let e=c();return e.o||(e.o={ST:n.gm.setTimeout,SI:n.gm.setImmediate||n.gm.setInterval,CT:n.gm.clearTimeout,XHR:n.gm.XMLHttpRequest,REQ:n.gm.Request,EV:n.gm.Event,PR:n.gm.Promise,MO:n.gm.MutationObserver,FETCH:n.gm.fetch,WS:n.gm.WebSocket},(0,o.i)(...Object.values(e.o))),e}function u(e,t){let r=c();r.initializedAgents??={},t.initializedAt={ms:(0,i.t)(),date:new Date},r.initializedAgents[e]=t,2===Object.keys(r.initializedAgents).length&&(0,s.R)(69)}function l(){return function(){let e=c();const t=e.info||{};e.info={beacon:a.beacon,errorBeacon:a.errorBeacon,...t}}(),function(){let e=c();const t=e.init||{};e.init={...t}}(),d(),function(){let e=c();const t=e.loader_config||{};e.loader_config={...t}}(),c()}},782:(e,t,r)=>{"use strict";r.d(t,{T:()=>n});const n=r(860).K7.pageViewTiming},860:(e,t,r)=>{"use strict";r.d(t,{$J:()=>u,K7:()=>c,P3:()=>d,XX:()=>i,Yy:()=>a,df:()=>s,qY:()=>n,v4:()=>o});const n="events",i="jserrors",s="browser/blobs",o="rum",a="browser/logs",c={ajax:"ajax",genericEvents:"generic_events",jserrors:i,logging:"logging",metrics:"metrics",pageAction:"page_action",pageViewEvent:"page_view_event",pageViewTiming:"page_view_timing",sessionReplay:"session_replay",sessionTrace:"session_trace",softNav:"soft_navigations"},d={[c.pageViewEvent]:1,[c.pageViewTiming]:2,[c.metrics]:3,[c.jserrors]:4,[c.softNav]:5,[c.ajax]:6,[c.sessionTrace]:7,[c.sessionReplay]:8,[c.logging]:9,[c.genericEvents]:10},u={[c.pageViewEvent]:o,[c.pageViewTiming]:n,[c.ajax]:n,[c.softNav]:n,[c.metrics]:i,[c.jserrors]:i,[c.sessionTrace]:s,[c.sessionReplay]:s,[c.logging]:a,[c.genericEvents]:"ins"}},944:(e,t,r)=>{"use strict";r.d(t,{R:()=>i});var n=r(3241);function i(e,t){"function"==typeof console.debug&&(console.debug("New Relic Warning: https://github.com/newrelic/newrelic-browser-agent/blob/main/docs/warning-codes.md#".concat(e),t),(0,n.W)({drained:null,type:"data",name:"warn",feature:"warn",data:{code:e,secondary:t}}))}},993:(e,t,r)=>{"use strict";r.d(t,{A$:()=>s,ET:()=>o,TZ:()=>a,p_:()=>i});var n=r(860);const i={ERROR:"ERROR",WARN:"WARN",INFO:"INFO",DEBUG:"DEBUG",TRACE:"TRACE"},s={OFF:0,ERROR:1,WARN:2,INFO:3,DEBUG:4,TRACE:5},o="log",a=n.K7.logging},1687:(e,t,r)=>{"use strict";r.d(t,{Ak:()=>a,Ze:()=>d,x3:()=>c});var n=r(3241),i=r(3606),s=r(860),o=r(2646);function a(e,t){if(!e)return;const r={staged:!1,priority:s.P3[t]||0};e.runtime.drainRegistry.get(t)||e.runtime.drainRegistry.set(t,r)}function c(e,t){if(!e)return;const r=e.runtime.drainRegistry;r&&(r.get(t)&&r.delete(t),l(e,t,!1),r.size&&u(e))}function d(e,t="feature",r=!1){if(e){if(!e.runtime.drainRegistry.get(t)||r)return l(e,t);e.runtime.drainRegistry.get(t).staged=!0,u(e)}}function u(e){if(!e)return;const t=Array.from(e.runtime.drainRegistry);t.every(([e,t])=>t.staged)&&(t.sort((e,t)=>e[1].priority-t[1].priority),t.forEach(([t])=>{e.runtime.drainRegistry.delete(t),l(e,t)}))}function l(e,t,r=!0){if(!e)return;const s=e.ee,a=i.i.handlers;if(s&&!s.aborted&&s.backlog&&a){if((0,n.W)({type:"lifecycle",name:"drain",feature:t}),r){const e=s.backlog[t],r=a[t];if(r){for(let t=0;e&&t<e.length;++t)f(e[t],r);Object.entries(r).forEach(([e,t])=>{Object.values(t||{}).forEach(t=>{t[0]?.on&&t[0].context()instanceof o.y&&!t[0].listeners(e).includes(t[1])&&t[0].on(e,t[1])})})}}s.isolatedBacklog||delete a[t],s.backlog[t]=null,s.emit("drain-"+t,[])}}function f(e,t){var r=e[1];Object.values(t[r]||{}).forEach(t=>{var r=e[0];if(t[0]===r){var n=t[1],i=e[3],s=e[2];n.apply(i,s)}})}},1738:(e,t,r)=>{"use strict";r.d(t,{U:()=>f,Y:()=>l});var n=r(3241),i=r(9908),s=r(1863),o=r(944),a=r(3969),c=r(8362),d=r(860),u=r(4261);function l(e,t,r,s){const l=s||r;!l||l[e]&&l[e]!==c.d.prototype[e]||(l[e]=function(){(0,i.p)(a.xV,["API/"+e+"/called"],void 0,d.K7.metrics,r.ee),(0,n.W)({drained:!!r.runtime?.activatedFeatures,type:"data",name:"api",feature:u.Pl+e,data:{}});try{return t.apply(this,arguments)}catch(e){(0,o.R)(23,e)}})}function f(e,t,r,n,o){const a=e.info;null===r?delete a.jsAttributes[t]:a.jsAttributes[t]=r,(o||null===r)&&(0,i.p)(u.Pl+n,[(0,s.t)(),t,r],void 0,"session",e.ee)}},1741:(e,t,r)=>{"use strict";r.d(t,{W:()=>s});var n=r(944),i=r(4261);class s{#e(e,...t){if(this[e]!==s.prototype[e])return this[e](...t);(0,n.R)(35,e)}addPageAction(e,t){return this.#e(i.hG,e,t)}register(e){return this.#e(i.eY,e)}recordCustomEvent(e,t){return this.#e(i.fF,e,t)}setPageViewName(e,t){return this.#e(i.Fw,e,t)}setCustomAttribute(e,t,r){return this.#e(i.cD,e,t,r)}noticeError(e,t){return this.#e(i.o5,e,t)}setUserId(e,t=!1){return this.#e(i.Dl,e,t)}setApplicationVersion(e){return this.#e(i.nb,e)}setErrorHandler(e){return this.#e(i.bt,e)}addRelease(e,t){return this.#e(i.k6,e,t)}log(e,t){return this.#e(i.$9,e,t)}start(){return this.#e(i.d3)}finished(e){return this.#e(i.BL,e)}recordReplay(){return this.#e(i.CH)}pauseReplay(){return this.#e(i.Tb)}addToTrace(e){return this.#e(i.U2,e)}setCurrentRouteName(e){return this.#e(i.PA,e)}interaction(e){return this.#e(i.dT,e)}wrapLogger(e,t,r){return this.#e(i.Wb,e,t,r)}measure(e,t){return this.#e(i.V1,e,t)}consent(e){return this.#e(i.Pv,e)}}},1863:(e,t,r)=>{"use strict";function n(){return Math.floor(performance.now())}r.d(t,{t:()=>n})},1910:(e,t,r)=>{"use strict";r.d(t,{i:()=>s});var n=r(944);const i=new Map;function s(...e){return e.every(e=>{if(i.has(e))return i.get(e);const t="function"==typeof e?e.toString():"",r=t.includes("[native code]"),s=t.includes("nrWrapper");return r||s||(0,n.R)(64,e?.name||t),i.set(e,r),r})}},2555:(e,t,r)=>{"use strict";r.d(t,{D:()=>a,f:()=>o});var n=r(384),i=r(8122);const s={beacon:n.NT.beacon,errorBeacon:n.NT.errorBeacon,licenseKey:void 0,applicationID:void 0,sa:void 0,queueTime:void 0,applicationTime:void 0,ttGuid:void 0,user:void 0,account:void 0,product:void 0,extra:void 0,jsAttributes:{},userAttributes:void 0,atts:void 0,transactionName:void 0,tNamePlain:void 0};function o(e){try{return!!e.licenseKey&&!!e.errorBeacon&&!!e.applicationID}catch(e){return!1}}const a=e=>(0,i.a)(e,s)},2614:(e,t,r)=>{"use strict";r.d(t,{BB:()=>o,H3:()=>n,g:()=>d,iL:()=>c,tS:()=>a,uh:()=>i,wk:()=>s});const n="NRBA",i="SESSION",s=144e5,o=18e5,a={STARTED:"session-started",PAUSE:"session-pause",RESET:"session-reset",RESUME:"session-resume",UPDATE:"session-update"},c={SAME_TAB:"same-tab",CROSS_TAB:"cross-tab"},d={OFF:0,FULL:1,ERROR:2}},2646:(e,t,r)=>{"use strict";r.d(t,{y:()=>n});class n{constructor(e){this.contextId=e}}},2843:(e,t,r)=>{"use strict";r.d(t,{G:()=>s,u:()=>i});var n=r(3878);function i(e,t=!1,r,i){(0,n.DD)("visibilitychange",function(){if(t)return void("hidden"===document.visibilityState&&e());e(document.visibilityState)},r,i)}function s(e,t,r){(0,n.sp)("pagehide",e,t,r)}},3241:(e,t,r)=>{"use strict";r.d(t,{W:()=>s});var n=r(6154);const i="newrelic";function s(e={}){try{n.gm.dispatchEvent(new CustomEvent(i,{detail:e}))}catch(e){}}},3304:(e,t,r)=>{"use strict";r.d(t,{A:()=>s});var n=r(7836);const i=()=>{const e=new WeakSet;return(t,r)=>{if("object"==typeof r&&null!==r){if(e.has(r))return;e.add(r)}return r}};function s(e){try{return JSON.stringify(e,i())??""}catch(e){try{n.ee.emit("internal-error",[e])}catch(e){}return""}}},3333:(e,t,r)=>{"use strict";r.d(t,{$v:()=>u,TZ:()=>n,Xh:()=>c,Zp:()=>i,kd:()=>d,mq:()=>a,nf:()=>o,qN:()=>s});const n=r(860).K7.genericEvents,i=["auxclick","click","copy","keydown","paste","scrollend"],s=["focus","blur"],o=4,a=1e3,c=2e3,d=["PageAction","UserAction","BrowserPerformance"],u={RESOURCES:"experimental.resources",REGISTER:"register"}},3434:(e,t,r)=>{"use strict";r.d(t,{Jt:()=>o,YM:()=>u});var n=r(7836),i=r(5607),s=r(5732);const o="nr@original:".concat(i.W),a=50;var c=Object.prototype.hasOwnProperty,d=!1;function u(e,t,r){return e||(e=n.ee),i.inPlace=function(e,t,r,n,s,o){r||(r="");const a="-"===r.charAt(0);for(let c=0;c<t.length;c++){const d=t[c],u=e[d];f(u)||(e[d]=i(u,a?d+r:r,n,d,s,o))}},i.flag=o,i;function i(t,n,i,d,h,p){return f(t)?t:(n||(n=""),nrWrapper[o]=t,function(e,t,r){if(Object.defineProperty&&Object.keys)try{return Object.keys(e).forEach(function(r){Object.defineProperty(t,r,{get:function(){return e[r]},set:function(t){return e[r]=t,t}})}),t}catch(e){l([e],r)}for(var n in e)c.call(e,n)&&(t[n]=e[n])}(t,nrWrapper,e),nrWrapper);function nrWrapper(){var o,c,f,g;let m,v;try{c=this,o=[...arguments],v=p?(0,s.$5)(r):[void 0],f="function"==typeof i?i(o,c):i||{}}catch(t){l([t,"",[o,c,d],f],e)}u(n+"start",[o,c,d,v],f,h);const y=performance.now();let b;try{return g=t.apply(c,o),b=performance.now(),g}catch(e){throw b=performance.now(),u(n+"err",[o,c,e,v],f,h),m=e,m}finally{const e=b-y,t={start:y,end:b,duration:e,isLongTask:e>=a,methodName:d,thrownError:m};t.isLongTask&&u("long-task",[t,c,v],f,h),u(n+"end",[o,c,g,v],f,h)}}}function u(r,n,i,s){if(!d||t){var o=d;d=!0;try{e.emit(r,n,i,t,s)}catch(t){l([t,r,n,i],e)}d=o}}}function l(e,t){t||(t=n.ee);try{t.emit("internal-error",e)}catch(e){}}function f(e){return!(e&&"function"==typeof e&&e.apply&&!e[o])}},3606:(e,t,r)=>{"use strict";r.d(t,{i:()=>s});var n=r(9908);s.on=o;var i=s.handlers={};function s(e,t,r,s){o(s||n.d,i,e,t,r)}function o(e,t,r,i,s){s||(s="feature"),e||(e=n.d);var o=t[s]=t[s]||{};(o[r]=o[r]||[]).push([e,i])}},3738:(e,t,r)=>{"use strict";r.d(t,{He:()=>i,Kp:()=>a,Lc:()=>d,Rz:()=>u,TZ:()=>n,bD:()=>s,d3:()=>o,jx:()=>l,sl:()=>f,uP:()=>c});const n=r(860).K7.sessionTrace,i="bstResource",s="resource",o="-start",a="-end",c="fn"+o,d="fn"+a,u="pushState",l=1e3,f=3e4},3785:(e,t,r)=>{"use strict";r.d(t,{R:()=>c,b:()=>d});var n=r(9908),i=r(1863),s=r(860),o=r(3969),a=r(993);function c(e,t,r={},c=a.p_.INFO,d=!0,u,l=(0,i.t)()){(0,n.p)(o.xV,["API/logging/".concat(c.toLowerCase(),"/called")],void 0,s.K7.metrics,e),(0,n.p)(a.ET,[l,t,r,c,d,u],void 0,s.K7.logging,e)}function d(e){return"string"==typeof e&&Object.values(a.p_).some(t=>t===e.toUpperCase().trim())}},3878:(e,t,r)=>{"use strict";function n(e,t){return{capture:e,passive:!1,signal:t}}function i(e,t,r=!1,i){window.addEventListener(e,t,n(r,i))}function s(e,t,r=!1,i){document.addEventListener(e,t,n(r,i))}r.d(t,{DD:()=>s,jT:()=>n,sp:()=>i})},3962:(e,t,r)=>{"use strict";r.d(t,{AM:()=>o,O2:()=>l,OV:()=>s,Qu:()=>f,TZ:()=>c,ih:()=>h,pP:()=>a,t1:()=>u,tC:()=>i,wD:()=>d});var n=r(860);const i=["click","keydown","submit"],s="popstate",o="api",a="initialPageLoad",c=n.K7.softNav,d=5e3,u=500,l={INITIAL_PAGE_LOAD:"",ROUTE_CHANGE:1,UNSPECIFIED:2},f={INTERACTION:1,AJAX:2,CUSTOM_END:3,CUSTOM_TRACER:4},h={IP:"in progress",PF:"pending finish",FIN:"finished",CAN:"cancelled"}},3969:(e,t,r)=>{"use strict";r.d(t,{TZ:()=>n,XG:()=>a,rs:()=>i,xV:()=>o,z_:()=>s});const n=r(860).K7.metrics,i="sm",s="cm",o="storeSupportabilityMetrics",a="storeEventMetrics"},4234:(e,t,r)=>{"use strict";r.d(t,{W:()=>i});var n=r(1687);class i{constructor(e,t){this.agentRef=e,this.ee=e?.ee,this.featureName=t,this.blocked=!1}deregisterDrain(){(0,n.x3)(this.agentRef,this.featureName)}}},4261:(e,t,r)=>{"use strict";r.d(t,{$9:()=>u,BL:()=>c,CH:()=>p,Dl:()=>R,Fw:()=>w,PA:()=>v,Pl:()=>n,Pv:()=>x,Tb:()=>f,U2:()=>o,V1:()=>A,Wb:()=>T,bt:()=>b,cD:()=>y,d3:()=>E,dT:()=>d,eY:()=>g,fF:()=>h,hG:()=>s,hw:()=>i,k6:()=>a,nb:()=>m,o5:()=>l});const n="api-",i=n+"ixn-",s="addPageAction",o="addToTrace",a="addRelease",c="finished",d="interaction",u="log",l="noticeError",f="pauseReplay",h="recordCustomEvent",p="recordReplay",g="register",m="setApplicationVersion",v="setCurrentRouteName",y="setCustomAttribute",b="setErrorHandler",w="setPageViewName",R="setUserId",E="start",T="wrapLogger",A="measure",x="consent"},5205:(e,t,r)=>{"use strict";r.d(t,{j:()=>x});var n=r(384),i=r(1741);var s=r(2555),o=r(3333);const a=e=>{if(!e||"string"!=typeof e)return!1;try{document.createDocumentFragment().querySelector(e)}catch{return!1}return!0};var c=r(2614),d=r(944),u=r(8122);const l="[data-nr-mask]",f=e=>(0,u.a)(e,(()=>{const e={feature_flags:[],experimental:{register:!1,resources:!1},mask_selector:"*",block_selector:"[data-nr-block]",mask_input_options:{color:!1,date:!1,"datetime-local":!1,email:!1,month:!1,number:!1,range:!1,search:!1,tel:!1,text:!1,time:!1,url:!1,week:!1,textarea:!1,select:!1,password:!0}};return{ajax:{deny_list:void 0,block_internal:!0,enabled:!0,autoStart:!0},api:{register:{get enabled(){return e.feature_flags.includes(o.$v.REGISTER)||e.experimental.register},set enabled(t){e.experimental.register=t},duplicate_data_to_container:!1}},browser_consent_mode:{enabled:!1},distributed_tracing:{enabled:void 0,exclude_newrelic_header:void 0,cors_use_newrelic_header:void 0,cors_use_tracecontext_headers:void 0,allowed_origins:void 0},get feature_flags(){return e.feature_flags},set feature_flags(t){e.feature_flags=t},generic_events:{enabled:!0,autoStart:!0},harvest:{interval:30},jserrors:{enabled:!0,autoStart:!0},logging:{enabled:!0,autoStart:!0},metrics:{enabled:!0,autoStart:!0},obfuscate:void 0,page_action:{enabled:!0},page_view_event:{enabled:!0,autoStart:!0},page_view_timing:{enabled:!0,autoStart:!0},performance:{capture_marks:!1,capture_measures:!1,capture_detail:!0,resources:{get enabled(){return e.feature_flags.includes(o.$v.RESOURCES)||e.experimental.resources},set enabled(t){e.experimental.resources=t},asset_types:[],first_party_domains:[],ignore_newrelic:!0}},privacy:{cookies_enabled:!0},proxy:{assets:void 0,beacon:void 0},session:{expiresMs:c.wk,inactiveMs:c.BB},session_replay:{autoStart:!0,enabled:!1,preload:!1,sampling_rate:10,error_sampling_rate:100,collect_fonts:!1,inline_images:!1,fix_stylesheets:!0,mask_all_inputs:!0,get mask_text_selector(){return e.mask_selector},set mask_text_selector(t){a(t)?e.mask_selector="".concat(t,",").concat(l):""===t||null===t?e.mask_selector=l:(0,d.R)(5,t)},get block_class(){return"nr-block"},get ignore_class(){return"nr-ignore"},get mask_text_class(){return"nr-mask"},get block_selector(){return e.block_selector},set block_selector(t){a(t)?e.block_selector+=",".concat(t):""!==t&&(0,d.R)(6,t)},get mask_input_options(){return e.mask_input_options},set mask_input_options(t){t&&"object"==typeof t?e.mask_input_options={...t,password:!0}:(0,d.R)(7,t)}},session_trace:{enabled:!0,autoStart:!0},soft_navigations:{enabled:!0,autoStart:!0},ssl:void 0,user_actions:{enabled:!0,elementAttributes:["id","className","tagName","type"]}}})());var h=r(6154),p=r(9324);let g=0;const m={buildEnv:p.F3,distMethod:p.Xs,version:p.xv,originTime:h.WN},v={consented:!1},y={activatedFeatures:void 0,appMetadata:{},configured:!1,get consented(){return this.session?.state?.consent||v.consented},set consented(e){v.consented=e},customTransaction:void 0,denyList:[],disabled:!1,drainRegistry:new Map,harvester:void 0,isolatedBacklog:!1,isRecording:!1,loaderType:void 0,maxBytes:3e4,obfuscator:void 0,onerror:void 0,ptid:void 0,releaseIds:{},session:void 0,timeKeeper:void 0,registeredEntities:[],jsAttributesMetadata:{bytes:0},get harvestCount(){return++g}},b=e=>{const t=(0,u.a)(e,y),r=Object.keys(m).reduce((e,t)=>(e[t]={value:m[t],writable:!1,configurable:!0,enumerable:!0},e),{});return Object.defineProperties(t,r)},w=e=>{const t=e.startsWith("http");e+="/",r.p=t?e:"https://"+e};var R=r(7836),E=r(3241);const T={accountID:void 0,trustKey:void 0,agentID:void 0,licenseKey:void 0,applicationID:void 0,xpid:void 0},A=e=>(0,u.a)(e,T);function x(e,t={},r,o){let{init:a,info:c,loader_config:d,runtime:u={},exposed:l=!0}=t;if(!c){const e=(0,n.pV)();a=e.init,c=e.info,d=e.loader_config}e.init=f(a||{}),e.loader_config=A(d||{}),c.jsAttributes??={},h.bv&&(c.jsAttributes.isWorker=!0),e.info=(0,s.D)(c);const p=e.init;e.runtime??=b(u),p.proxy.assets&&w(p.proxy.assets),e.runtime.configured||(Object.defineProperty(e,"beacons",{get:()=>[e.info.beacon,e.info.errorBeacon,e.init.proxy.assets,e.init.proxy.beacon].filter(Boolean)}),Object.defineProperty(e.runtime,"denyList",{get:()=>[...e.init.ajax.deny_list||[],...e.init.ajax.block_internal?e.beacons:[]]}),e.runtime.ptid=e.agentIdentifier,function(e){const t=(0,n.pV)();Object.getOwnPropertyNames(i.W.prototype).forEach(r=>{const n=i.W.prototype[r];if("function"!=typeof n||"constructor"===n)return;let s=t[r];e[r]&&!1!==e.exposed&&"micro-agent"!==e.runtime?.loaderType&&(t[r]=(...t)=>{const n=e[r](...t);return s?s(...t):n})})}(e),e.runtime.loaderType=r,e.ee=R.ee.get(e.agentIdentifier),e.exposed=l,(0,E.W)({drained:!!e.runtime.activatedFeatures,type:"lifecycle",name:"initialize",feature:void 0,data:e.config}),e.runtime.configured=!0)}},5270:(e,t,r)=>{"use strict";r.d(t,{Aw:()=>o,SR:()=>s,rF:()=>a});var n=r(384),i=r(7767);function s(e){return!!(0,n.dV)().o.MO&&(0,i.V)(e)&&!0===e?.session_trace.enabled}function o(e){return!0===e?.session_replay.preload&&s(e)}function a(e,t){try{if("string"==typeof t?.type){if("password"===t.type.toLowerCase())return"*".repeat(e?.length||0);if(void 0!==t?.dataset?.nrUnmask||t?.classList?.contains("nr-unmask"))return e}}catch(e){}return"string"==typeof e?e.replace(/[\S]/g,"*"):"*".repeat(e?.length||0)}},5289:(e,t,r)=>{"use strict";r.d(t,{GG:()=>o,Qr:()=>c,sB:()=>a});var n=r(3878),i=r(6389);function s(){return"undefined"==typeof document||"complete"===document.readyState}function o(e,t){if(s())return e();const r=(0,i.J)(e),o=setInterval(()=>{s()&&(clearInterval(o),r())},500);(0,n.sp)("load",r,t)}function a(e){if(s())return e();(0,n.DD)("DOMContentLoaded",e)}function c(e){if(s())return e();(0,n.sp)("popstate",e)}},5607:(e,t,r)=>{"use strict";r.d(t,{W:()=>n});const n=(0,r(9566).bz)()},5732:(e,t,r)=>{"use strict";r.d(t,{$5:()=>u,B5:()=>d,Ms:()=>s,Ux:()=>a,YA:()=>c,fQ:()=>i,yx:()=>o});var n=r(7508);const i={MFE:"MFE",BA:"BA"};function s(e,t){if(!e||!t?.init.api.register.enabled)return[];const r=t.runtime.registeredEntities;return r?.filter(t=>String(t.metadata.target.id)===String(e)).map(e=>e.metadata.target)||[]}function o(e,t){if(!e||!t?.init.api.register.enabled)return[];const r=t.runtime.registeredEntities;return r?.filter(t=>t.metadata.timings?.asset?.endsWith(e)).map(e=>e.metadata.target)||[]}function a(e,t){if(!l(t))return{};const r=t.agentRef.runtime.appMetadata.agents[0].entityGuid;return e?e.attributes:{"entity.guid":r,appId:t.agentRef.info.applicationID}}function c(e,t){return d(e,t)?{"child.id":e.id,"child.type":e.type,...a(void 0,t)}:{}}function d(e,t){return!!e&&!!l(t)&&t.agentRef.init.api.register.duplicate_data_to_container}function u(e){if(!e?.init.api.register.enabled)return[void 0];const t=[];try{var r=(0,n.AZ)((0,n.QL)());let i=r.length-1;for(;r[i];)t.push(...o(r[i--],e))}catch(e){}return t.length||t.push(void 0),t}function l(e){return 2===e?.harvestEndpointVersion}},6154:(e,t,r)=>{"use strict";r.d(t,{OF:()=>d,RI:()=>i,WN:()=>f,bv:()=>s,gm:()=>o,lR:()=>l,m:()=>c,mw:()=>a,sb:()=>u,zk:()=>h});var n=r(1863);const i="undefined"!=typeof window&&!!window.document,s="undefined"!=typeof WorkerGlobalScope&&("undefined"!=typeof self&&self instanceof WorkerGlobalScope&&self.navigator instanceof WorkerNavigator||"undefined"!=typeof globalThis&&globalThis instanceof WorkerGlobalScope&&globalThis.navigator instanceof WorkerNavigator),o=i?window:"undefined"!=typeof WorkerGlobalScope&&("undefined"!=typeof self&&self instanceof WorkerGlobalScope&&self||"undefined"!=typeof globalThis&&globalThis instanceof WorkerGlobalScope&&globalThis),a=Boolean("hidden"===o?.document?.visibilityState),c=""+o?.location,d=/iPad|iPhone|iPod/.test(o.navigator?.userAgent),u=d&&"undefined"==typeof SharedWorker,l=(()=>{const e=o.navigator?.userAgent?.match(/Firefox[/\s](\d+\.\d+)/);return Array.isArray(e)&&e.length>=2?+e[1]:0})(),f=Date.now()-(0,n.t)(),h=()=>{const e=o?.performance?.getEntriesByType?.("navigation")?.[0];if(e&&e.responseStart>0&&e.responseStart<o.performance.now())return e}},6344:(e,t,r)=>{"use strict";r.d(t,{BB:()=>u,Qb:()=>l,TZ:()=>i,Ug:()=>o,Vh:()=>s,_s:()=>a,bc:()=>d,yP:()=>c});var n=r(2614);const i=r(860).K7.sessionReplay,s="errorDuringReplay",o=.12,a={DomContentLoaded:0,Load:1,FullSnapshot:2,IncrementalSnapshot:3,Meta:4,Custom:5},c={[n.g.ERROR]:15e3,[n.g.FULL]:3e5,[n.g.OFF]:0},d={RESET:{message:"Session was reset",sm:"Reset"},IMPORT:{message:"Recorder failed to import",sm:"Import"},TOO_MANY:{message:"429: Too Many Requests",sm:"Too-Many"},TOO_BIG:{message:"Payload was too large",sm:"Too-Big"},CROSS_TAB:{message:"Session Entity was set to OFF on another tab",sm:"Cross-Tab"},ENTITLEMENTS:{message:"Session Replay is not allowed and will not be started",sm:"Entitlement"}},u=5e3,l={API:"api",RESUME:"resume",SWITCH_TO_FULL:"switchToFull",INITIALIZE:"initialize",PRELOAD:"preload"}},6389:(e,t,r)=>{"use strict";function n(e,t=500,r={}){const n=r?.leading||!1;let i;return(...r)=>{n&&void 0===i&&(e.apply(this,r),i=setTimeout(()=>{i=clearTimeout(i)},t)),n||(clearTimeout(i),i=setTimeout(()=>{e.apply(this,r)},t))}}function i(e){let t=!1;return(...r)=>{t||(t=!0,e.apply(this,r))}}r.d(t,{J:()=>i,s:()=>n})},6630:(e,t,r)=>{"use strict";r.d(t,{T:()=>n});const n=r(860).K7.pageViewEvent},6774:(e,t,r)=>{"use strict";r.d(t,{T:()=>n});const n=r(860).K7.jserrors},7295:(e,t,r)=>{"use strict";r.d(t,{Xv:()=>o,gX:()=>i,iW:()=>s});var n=[];function i(e){if(!e||s(e))return!1;if(0===n.length)return!0;if("*"===n[0].hostname)return!1;for(var t=0;t<n.length;t++){var r=n[t];if(r.hostname.test(e.hostname)&&r.pathname.test(e.pathname))return!1}return!0}function s(e){return void 0===e.hostname}function o(e){if(n=[],e&&e.length)for(var t=0;t<e.length;t++){let r=e[t];if(!r)continue;if("*"===r)return void(n=[{hostname:"*"}]);0===r.indexOf("http://")?r=r.substring(7):0===r.indexOf("https://")&&(r=r.substring(8));const i=r.indexOf("/");let s,o;i>0?(s=r.substring(0,i),o=r.substring(i)):(s=r,o="*");let[c]=s.split(":");n.push({hostname:a(c),pathname:a(o,!0)})}}function a(e,t=!1){const r=e.replace(/[.+?^${}()|[\]\\]/g,e=>"\\"+e).replace(/\*/g,".*?");return new RegExp((t?"^":"")+r+"$")}},7485:(e,t,r)=>{"use strict";r.d(t,{D:()=>i});var n=r(6154);function i(e){if(0===(e||"").indexOf("data:"))return{protocol:"data"};try{const t=new URL(e,location.href),r={port:t.port,hostname:t.hostname,pathname:t.pathname,search:t.search,protocol:t.protocol.slice(0,t.protocol.indexOf(":")),sameOrigin:t.protocol===n.gm?.location?.protocol&&t.host===n.gm?.location?.host};return r.port&&""!==r.port||("http:"===t.protocol&&(r.port="80"),"https:"===t.protocol&&(r.port="443")),r.pathname&&""!==r.pathname?r.pathname.startsWith("/")||(r.pathname="/".concat(r.pathname)):r.pathname="/",r}catch(e){return{}}}},7508:(e,t,r)=>{"use strict";r.d(t,{AZ:()=>g,Qr:()=>b,QL:()=>m});var n=r(6154),i=r(1863),s=r(9119),o=r(7866);class a{dom=new c;performance=new c;constructor(e){this.url=e}get script(){const e=Math.max(this.dom.start,this.performance.end);return{start:e,end:Math.max(this.dom.end,this.performance.end,e)}}}class c{start=0;end=0;value=void 0}let d;try{d=g(m())[0]}catch(e){d=g(e)[0]}const u=e=>"script"===e.initiatorType||["link","fetch"].includes(e.initiatorType)&&e.name.endsWith(".js"),l=new Map;let f=[];function h(e){return l.get(e)}function p(e){const t=h(e);if(t)return t;const r=new a(e);if(l.set(e,r),l.size>1e3){const e=l.keys().next().value;l.delete(e)}return r}if(n.gm.MutationObserver&&n.gm.document){new MutationObserver(e=>{e.forEach(e=>{e.addedNodes.forEach(e=>{if("SCRIPT"===e.nodeName&&e.src){const t=p((0,s.L)(e.src));t.dom.start=(0,i.t)(),t.dom.value=e;const r=()=>{t.dom.end=(0,i.t)()};["load","error"].forEach(t=>e.addEventListener(t,r,{once:!0}))}})})}).observe(n.gm.document,{childList:!0,subtree:!0})}if(n.gm.PerformanceObserver?.supportedEntryTypes.includes("resource")){new PerformanceObserver(e=>{e.getEntries().filter(u).forEach(e=>{const t=p((0,s.L)(e.name));t.performance.start=Math.floor(e.startTime),t.performance.end=Math.floor(e.responseEnd),t.performance.value=e;const r=[];f.forEach(({test:t,addedAt:n},s)=>{(t(e)||(0,i.t)()-n>1e4)&&r.push(s)}),f=f.filter((e,t)=>!r.includes(t))})}).observe({type:"resource",buffered:!0})}function g(e){if(!e||"string"!=typeof e)return[];const t=new Set,r=e.split("\n");for(const e of r){const r=e.match(o.cn)||e.match(o.hB)||e.match(o.fL);if(r&&r[2])t.add((0,s.L)(r[2]));else{const r=e.match(/\(([^)]+\.js):\d+:\d+\)/)||e.match(/^\s+at\s+([^\s(]+\.js):\d+:\d+/);r&&r[1]&&t.add((0,s.L)(r[1]))}}return[...t]}function m(){let e;try{const t=Error.stackTraceLimit;Error.stackTraceLimit=50,e=(new Error).stack,Error.stackTraceLimit=t}catch(t){e=(new Error).stack}return e}function v(e,t){return(0,s.L)(e.name)===t}function y(e,t){e.fetchStart=Math.floor(t.startTime),e.fetchEnd=Math.floor(t.responseEnd),e.asset=t.name,e.type=t.initiatorType}function b(){const e={registeredAt:(0,i.t)(),reportedAt:void 0,fetchStart:0,fetchEnd:0,scriptStart:0,scriptEnd:0,asset:void 0,type:"unknown"},t=m();if(!t)return e;const r=n.gm.performance?.getEntriesByType("navigation")?.[0]?.name||"";try{const o=g(t),a=(o.length>1?o.filter(e=>d!==e):o)[0];if(!a)return e;if(r.includes(a))return e.asset=(0,s.L)(r),e.type="inline",e;e.correlation=h(a);const c=e.correlation?.performance.value||performance.getEntriesByType("resource").find(e=>v(e,a));c?y(e,c):function(e){if(!e||!n.gm.document)return!1;try{const t=n.gm.document.querySelectorAll('link[rel="preload"][as="script"]');for(const r of t)if((0,s.L)(r.href)===e)return!0}catch(e){}return!1}(a)&&(e.asset=a,e.type="preload",f.push({addedAt:(0,i.t)(),test:t=>!!v(t,a)&&(y(e,t),!0)})),Object.defineProperty(e,"scriptStart",{get:()=>e.correlation?.script.start||e.fetchEnd}),Object.defineProperty(e,"scriptEnd",{get:()=>e.correlation?.script.end||e.registeredAt})}catch(e){}return e}},7699:(e,t,r)=>{"use strict";r.d(t,{It:()=>s,KC:()=>a,No:()=>i,qh:()=>o});var n=r(860);const i=16e3,s=1e6,o="SESSION_ERROR",a={[n.K7.logging]:!0,[n.K7.genericEvents]:!0,[n.K7.jserrors]:!0,[n.K7.ajax]:!0}},7767:(e,t,r)=>{"use strict";r.d(t,{V:()=>i});var n=r(6154);const i=e=>n.RI&&!0===e?.privacy.cookies_enabled},7836:(e,t,r)=>{"use strict";r.d(t,{P:()=>a,ee:()=>c});var n=r(384),i=r(8990),s=r(2646),o=r(5607);const a="nr@context:".concat(o.W),c=function e(t,r){var n={},o={},u={},l=!1;try{l=16===r.length&&d.initializedAgents?.[r]?.runtime.isolatedBacklog}catch(e){}var f={on:p,addEventListener:p,removeEventListener:function(e,t){var r=n[e];if(!r)return;for(var i=0;i<r.length;i++)r[i]===t&&r.splice(i,1)},emit:function(e,r,n,i,s){!1!==s&&(s=!0);if(c.aborted&&!i)return;t&&s&&t.emit(e,r,n);var a=h(n);g(e).forEach(e=>{e.apply(a,r)});var d=v()[o[e]];d&&d.push([f,e,r,a]);return a},get:m,listeners:g,context:h,buffer:function(e,t){const r=v();if(t=t||"feature",f.aborted)return;Object.entries(e||{}).forEach(([e,n])=>{o[n]=t,t in r||(r[t]=[])})},abort:function(){f._aborted=!0,Object.keys(f.backlog).forEach(e=>{delete f.backlog[e]})},isBuffering:function(e){return!!v()[o[e]]},debugId:r,backlog:l?{}:t&&"object"==typeof t.backlog?t.backlog:{},isolatedBacklog:l};return Object.defineProperty(f,"aborted",{get:()=>{let e=f._aborted||!1;return e||(t&&(e=t.aborted),e)}}),f;function h(e){return e&&e instanceof s.y?e:e?(0,i.I)(e,a,()=>new s.y(a)):new s.y(a)}function p(e,t){n[e]=g(e).concat(t)}function g(e){return n[e]||[]}function m(t){return u[t]=u[t]||e(f,t)}function v(){return f.backlog}}(void 0,"globalEE"),d=(0,n.Zm)();d.ee||(d.ee=c)},7866:(e,t,r)=>{"use strict";r.d(t,{Nc:()=>s,cn:()=>a,fL:()=>i,h3:()=>n,hB:()=>o});const n=/function (.+?)\s*\(/,i=/^\s*at .+ \(eval at \S+ \((?:(?:file|http|https):[^)]+)?\)(?:, [^:]*:\d+:\d+)?\)$/i,s=/^\s*at Function code \(Function code:\d+:\d+\)\s*/i,o=/^\s*at (?:((?:\[object object\])?(?:[^(]*\([^)]*\))*[^()]*(?: \[as \S+\])?) )?\(?((?:file|http|https|chrome-extension):.*?)?:(\d+)(?::(\d+))?\)?\s*$/i,a=/^\s*(?:([^@]*)(?:\(.*?\))?@)?((?:file|http|https|chrome|safari-extension).*?):(\d+)(?::(\d+))?\s*$/i},8122:(e,t,r)=>{"use strict";r.d(t,{a:()=>i});var n=r(944);function i(e,t){try{if(!e||"object"!=typeof e)return(0,n.R)(3);if(!t||"object"!=typeof t)return(0,n.R)(4);const r=Object.create(Object.getPrototypeOf(t),Object.getOwnPropertyDescriptors(t)),s=0===Object.keys(r).length?e:r;for(let o in s)if(void 0!==e[o])try{if(null===e[o]){r[o]=null;continue}Array.isArray(e[o])&&Array.isArray(t[o])?r[o]=Array.from(new Set([...e[o],...t[o]])):e[o]instanceof Map||e[o]instanceof Set||e[o]instanceof Date||e[o]instanceof RegExp?r[o]=e[o]:"object"==typeof e[o]&&"object"==typeof t[o]?r[o]=i(e[o],t[o]):r[o]=e[o]}catch(e){r[o]||(0,n.R)(1,e)}return r}catch(e){(0,n.R)(2,e)}}},8139:(e,t,r)=>{"use strict";r.d(t,{u:()=>f});var n=r(7836),i=r(3434),s=r(8990),o=r(6154);const a={},c=o.gm.XMLHttpRequest,d="addEventListener",u="removeEventListener",l="nr@wrapped:".concat(n.P);function f(e){var t=function(e){return(e||n.ee).get("events")}(e);if(a[t.debugId]++)return t;a[t.debugId]=1;var r=(0,i.YM)(t,!0);function f(e){r.inPlace(e,[d,u],"-",p)}function p(e,t){return e[1]}return"getPrototypeOf"in Object&&(o.RI&&h(document,f),c&&h(c.prototype,f),h(o.gm,f)),t.on(d+"-start",function(e,t){var n=e[1];if(null!==n&&("function"==typeof n||"object"==typeof n)&&"newrelic"!==e[0]){var i=(0,s.I)(n,l,function(){var e={object:function(){if("function"!=typeof n.handleEvent)return;return n.handleEvent.apply(n,arguments)},function:n}[typeof n];return e?r(e,"fn-",null,e.name||"anonymous"):n});this.wrapped=e[1]=i}}),t.on(u+"-start",function(e){e[1]=this.wrapped||e[1]}),t}function h(e,t,...r){let n=e;for(;"object"==typeof n&&!Object.prototype.hasOwnProperty.call(n,d);)n=Object.getPrototypeOf(n);n&&t(n,...r)}},8362:(e,t,r)=>{"use strict";r.d(t,{d:()=>s});var n=r(9566),i=r(1741);class s extends i.W{agentIdentifier=(0,n.LA)(16)}},8374:(e,t,r)=>{r.nc=(()=>{try{return document?.currentScript?.nonce}catch(e){}return""})()},8990:(e,t,r)=>{"use strict";r.d(t,{I:()=>i});var n=Object.prototype.hasOwnProperty;function i(e,t,r){if(n.call(e,t))return e[t];var i=r();if(Object.defineProperty&&Object.keys)try{return Object.defineProperty(e,t,{value:i,writable:!0,enumerable:!1}),i}catch(e){}return e[t]=i,i}},9119:(e,t,r)=>{"use strict";r.d(t,{L:()=>s});var n=/([^?#]*)[^#]*(#[^?]*|$).*/,i=/([^?#]*)().*/;function s(e,t){return e?e.replace(t?n:i,"$1$2"):e}},9300:(e,t,r)=>{"use strict";r.d(t,{T:()=>n,f:()=>i});const n=r(860).K7.ajax,i="ajaxRequest.id"},9324:(e,t,r)=>{"use strict";r.d(t,{AJ:()=>o,F3:()=>i,Xs:()=>s,Yq:()=>a,xv:()=>n});const n="1.314.0",i="PROD",s="CDN",o="@newrelic/rrweb",a="1.1.0"},9566:(e,t,r)=>{"use strict";r.d(t,{LA:()=>a,ZF:()=>c,bz:()=>o,el:()=>d});var n=r(6154);const i="xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx";function s(e,t){return e?15&e[t]:16*Math.random()|0}function o(){const e=n.gm?.crypto||n.gm?.msCrypto;let t,r=0;return e&&e.getRandomValues&&(t=e.getRandomValues(new Uint8Array(30))),i.split("").map(e=>"x"===e?s(t,r++).toString(16):"y"===e?(3&s()|8).toString(16):e).join("")}function a(e){const t=n.gm?.crypto||n.gm?.msCrypto;let r,i=0;t&&t.getRandomValues&&(r=t.getRandomValues(new Uint8Array(e)));const o=[];for(var a=0;a<e;a++)o.push(s(r,i++).toString(16));return o.join("")}function c(){return a(16)}function d(){return a(32)}},9908:(e,t,r)=>{"use strict";r.d(t,{d:()=>n,p:()=>i});var n=r(7836).ee.get("handle");function i(e,t,r,i,s){s?(s.buffer([e],i),s.emit(e,t,r)):(n.buffer([e],i),n.emit(e,t,r))}}},n={};function i(e){var t=n[e];if(void 0!==t)return t.exports;var s=n[e]={exports:{}};return r[e](s,s.exports,i),s.exports}i.m=r,i.d=(e,t)=>{for(var r in t)i.o(t,r)&&!i.o(e,r)&&Object.defineProperty(e,r,{enumerable:!0,get:t[r]})},i.f={},i.e=e=>Promise.all(Object.keys(i.f).reduce((t,r)=>(i.f[r](e,t),t),[])),i.u=e=>({212:"nr-spa-compressor",249:"nr-spa-recorder",478:"nr-spa"}[e]+"-1.314.0.min.js"),i.o=(e,t)=>Object.prototype.hasOwnProperty.call(e,t),e={},t="NRBA-1.314.0.PROD:",i.l=(r,n,s,o)=>{if(e[r])e[r].push(n);else{var a,c;if(void 0!==s)for(var d=document.getElementsByTagName("script"),u=0;u<d.length;u++){var l=d[u];if(l.getAttribute("src")==r||l.getAttribute("data-webpack")==t+s){a=l;break}}if(!a){c=!0;var f={478:"sha512-mgj9qD5BdNJpEJCPmd1OfPzfnL/HRJ64OPYq/GZQxKc0756ytnQixcg8IA8iZWe/PRWN4DuGAnX82J2lscbiIw==",249:"sha512-fUJXIKjSt6UL9n1UZzNMzGGL3CxnBT/AgtoTat0/ONINbEB2PoiO+yK54DPYDwzpXDBddJHvs2BrYFj1GBpo2Q==",212:"sha512-vjoeBTgR0ONx6WcgBdfuPgcta1834zMPDhLL05C9c/pB+/xAuxL4BJMu9yeXYh+oIV7oIn6qTkDC7rBcZIDiHg=="};(a=document.createElement("script")).charset="utf-8",i.nc&&a.setAttribute("nonce",i.nc),a.setAttribute("data-webpack",t+s),a.src=r,0!==a.src.indexOf(window.location.origin+"/")&&(a.crossOrigin="anonymous"),f[o]&&(a.integrity=f[o])}e[r]=[n];var h=(t,n)=>{a.onerror=a.onload=null,clearTimeout(p);var i=e[r];if(delete e[r],a.parentNode&&a.parentNode.removeChild(a),i&&i.forEach(e=>e(n)),t)return t(n)},p=setTimeout(h.bind(null,void 0,{type:"timeout",target:a}),12e4);a.onerror=h.bind(null,a.onerror),a.onload=h.bind(null,a.onload),c&&document.head.appendChild(a)}},i.r=e=>{"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},i.p="https://js-agent.newrelic.com/",(()=>{var e={38:0,788:0};i.f.j=(t,r)=>{var n=i.o(e,t)?e[t]:void 0;if(0!==n)if(n)r.push(n[2]);else{var s=new Promise((r,i)=>n=e[t]=[r,i]);r.push(n[2]=s);var o=i.p+i.u(t),a=new Error;i.l(o,r=>{if(i.o(e,t)&&(0!==(n=e[t])&&(e[t]=void 0),n)){var s=r&&("load"===r.type?"missing":r.type),o=r&&r.target&&r.target.src;a.message="Loading chunk "+t+" failed: ("+s+": "+o+")",a.name="ChunkLoadError",a.type=s,a.request=o,n[1](a)}},"chunk-"+t,t)}};var t=(t,r)=>{var n,s,[o,a,c]=r,d=0;if(o.some(t=>0!==e[t])){for(n in a)i.o(a,n)&&(i.m[n]=a[n]);if(c)c(i)}for(t&&t(r);d<o.length;d++)s=o[d],i.o(e,s)&&e[s]&&e[s][0](),e[s]=0},r=self["webpackChunk:NRBA-1.314.0.PROD"]=self["webpackChunk:NRBA-1.314.0.PROD"]||[];r.forEach(t.bind(null,0)),r.push=t.bind(null,r.push.bind(r))})(),(()=>{"use strict";i(8374);var e=i(8362),t=i(860);const r=Object.values(t.K7);var n=i(5205);var s=i(9908),o=i(1863),a=i(4261),c=i(1738);var d=i(1687),u=i(4234),l=i(5289),f=i(6154),h=i(944),p=i(5270),g=i(7767),m=i(6389),v=i(7699);class y extends u.W{constructor(e,t){super(e,t),this.abortHandler=void 0,this.featAggregate=void 0,this.loadedSuccessfully=void 0,this.onAggregateImported=new Promise(e=>{this.loadedSuccessfully=e}),this.deferred=Promise.resolve(),!1===e.init[this.featureName].autoStart?this.deferred=new Promise((t,r)=>{this.ee.on("manual-start-all",(0,m.J)(()=>{(0,d.Ak)(e,this.featureName),t()}))}):(0,d.Ak)(e,t)}importAggregator(e,t,r={}){if(this.featAggregate)return;const n=async()=>{let n;await this.deferred;try{if((0,g.V)(e.init)){const{setupAgentSession:t}=await i.e(478).then(i.bind(i,8766));n=t(e)}}catch(e){(0,h.R)(20,e),this.ee.emit("internal-error",[e]),(0,s.p)(v.qh,[e],void 0,this.featureName,this.ee)}try{if(!this.#t(this.featureName,n,e.init))return(0,d.Ze)(this.agentRef,this.featureName),void this.loadedSuccessfully(!1);const{Aggregate:i}=await t();this.featAggregate=new i(e,r),e.runtime.harvester.initializedAggregates.push(this.featAggregate),this.loadedSuccessfully(!0)}catch(e){(0,h.R)(34,e),this.abortHandler?.(),(0,d.Ze)(this.agentRef,this.featureName,!0),this.loadedSuccessfully(!1),this.ee&&this.ee.abort()}};f.RI?(0,l.GG)(()=>n(),!0):n()}#t(e,r,n){if(this.blocked)return!1;switch(e){case t.K7.sessionReplay:return(0,p.SR)(n)&&!!r;case t.K7.sessionTrace:return!!r;default:return!0}}}var b=i(6630),w=i(2614),R=i(3241);class E extends y{static featureName=b.T;constructor(e){var t;super(e,b.T),this.setupInspectionEvents(),t=e,(0,c.Y)(a.Fw,function(e,r){"string"==typeof e&&("/"!==e.charAt(0)&&(e="/"+e),t.runtime.customTransaction=(r||"http://custom.transaction")+e,(0,s.p)(a.Pl+a.Fw,[(0,o.t)()],void 0,void 0,t.ee))},t),this.importAggregator(e,()=>i.e(478).then(i.bind(i,5839)))}setupInspectionEvents(){const e=(e,t)=>{e&&(0,R.W)({timeStamp:e.timeStamp,loaded:"complete"===e.target.readyState,type:"window",name:t,data:e.target.location+""})};(0,l.sB)(t=>{e(t,"DOMContentLoaded")}),(0,l.GG)(t=>{e(t,"load")}),(0,l.Qr)(t=>{e(t,"navigate")}),this.ee.on(w.tS.UPDATE,(e,t)=>{(0,R.W)({type:"lifecycle",name:"session",data:t})})}}var T=i(384);class A extends e.d{constructor(e){var t;(super(),f.gm)?(this.features={},(0,T.bQ)(this.agentIdentifier,this),this.desiredFeatures=new Set(e.features||[]),this.desiredFeatures.add(E),(0,n.j)(this,e,e.loaderType||"agent"),t=this,(0,c.Y)(a.cD,function(e,r,n=!1){if("string"==typeof e){if(["string","number","boolean"].includes(typeof r)||null===r)return(0,c.U)(t,e,r,a.cD,n);(0,h.R)(40,typeof r)}else(0,h.R)(39,typeof e)},t),function(e){(0,c.Y)(a.Dl,function(t,r=!1){if("string"!=typeof t&&null!==t)return void(0,h.R)(41,typeof t);const n=e.info.jsAttributes["enduser.id"];r&&null!=n&&n!==t?(0,s.p)(a.Pl+"setUserIdAndResetSession",[t],void 0,"session",e.ee):(0,c.U)(e,"enduser.id",t,a.Dl,!0)},e)}(this),function(e){(0,c.Y)(a.nb,function(t){if("string"==typeof t||null===t)return(0,c.U)(e,"application.version",t,a.nb,!1);(0,h.R)(42,typeof t)},e)}(this),function(e){(0,c.Y)(a.d3,function(){e.ee.emit("manual-start-all")},e)}(this),function(e){(0,c.Y)(a.Pv,function(t=!0){if("boolean"==typeof t){if((0,s.p)(a.Pl+a.Pv,[t],void 0,"session",e.ee),e.runtime.consented=t,t){const t=e.features.page_view_event;t.onAggregateImported.then(e=>{const r=t.featAggregate;e&&!r.sentRum&&r.sendRum()})}}else(0,h.R)(65,typeof t)},e)}(this),this.run()):(0,h.R)(21)}get config(){return{info:this.info,init:this.init,loader_config:this.loader_config,runtime:this.runtime}}get api(){return this}run(){try{const e=function(e){const t={};return r.forEach(r=>{t[r]=!!e[r]?.enabled}),t}(this.init),n=[...this.desiredFeatures];n.sort((e,r)=>t.P3[e.featureName]-t.P3[r.featureName]),n.forEach(r=>{if(!e[r.featureName]&&r.featureName!==t.K7.pageViewEvent)return;const n=function(e){switch(e){case t.K7.ajax:return[t.K7.jserrors];case t.K7.sessionTrace:return[t.K7.ajax,t.K7.pageViewEvent];case t.K7.sessionReplay:return[t.K7.sessionTrace];case t.K7.pageViewTiming:return[t.K7.pageViewEvent];default:return[]}}(r.featureName).filter(e=>!(e in this.features));n.length>0&&(0,h.R)(36,{targetFeature:r.featureName,missingDependencies:n}),this.features[r.featureName]=new r(this)})}catch(e){(0,h.R)(22,e);for(const e in this.features)this.features[e].abortHandler?.();const t=(0,T.Zm)();delete t.initializedAgents[this.agentIdentifier]?.features,delete this.sharedAggregator;return t.ee.get(this.agentIdentifier).abort(),!1}}}var x=i(2843),S=i(782);class _ extends y{static featureName=S.T;constructor(e){super(e,S.T),f.RI&&((0,x.u)(()=>(0,s.p)("docHidden",[(0,o.t)()],void 0,S.T,this.ee),!0),(0,x.G)(()=>(0,s.p)("winPagehide",[(0,o.t)()],void 0,S.T,this.ee)),this.importAggregator(e,()=>i.e(478).then(i.bind(i,9917))))}}var O=i(3969);class P extends y{static featureName=O.TZ;constructor(e){super(e,O.TZ),this.importAggregator(e,()=>i.e(478).then(i.bind(i,6555)))}}var k=i(6774),N=i(3878),j=i(3304);class D{constructor(e,t,r,n,i){this.name="UncaughtError",this.message="string"==typeof e?e:(0,j.A)(e),this.sourceURL=t,this.line=r,this.column=n,this.__newrelic=i}}function C(e){return M(e)?e:new D(void 0!==e?.message?e.message:e,e?.filename||e?.sourceURL,e?.lineno||e?.line,e?.colno||e?.col,e?.__newrelic,e?.cause)}function L(e){const t="Unhandled Promise Rejection: ";if(!e?.reason)return;if(M(e.reason)){try{e.reason.message.startsWith(t)||(e.reason.message=t+e.reason.message)}catch(e){}return C(e.reason)}const r=C(e.reason);return(r.message||"").startsWith(t)||(r.message=t+r.message),r}function I(e){if(e.error instanceof SyntaxError&&!/:\d+$/.test(e.error.stack?.trim())){const t=new D(e.message,e.filename,e.lineno,e.colno,e.error.__newrelic,e.cause);return t.name=SyntaxError.name,t}return M(e.error)?e.error:C(e)}function M(e){return e instanceof Error&&!!e.stack}function B(e,r,n,i,a=(0,o.t)()){"string"==typeof e&&(e=new Error(e)),(0,s.p)("err",[e,a,!1,r,n.runtime.isRecording,void 0,i],void 0,t.K7.jserrors,n.ee),(0,s.p)("uaErr",[],void 0,t.K7.genericEvents,n.ee)}var H=i(5732),K=i(993),W=i(3785);function F(e,{customAttributes:t={},level:r=K.p_.INFO}={},n,i,s=(0,o.t)()){(0,W.R)(n.ee,e,t,r,!1,i,s)}function U(e,r,n,i,c=(0,o.t)()){(0,s.p)(a.Pl+a.hG,[c,e,r,i],void 0,t.K7.genericEvents,n.ee)}function V(e,r,n,i,c=(0,o.t)()){const{start:d,end:u,customAttributes:l}=r||{},f={customAttributes:l||{}};if("object"!=typeof f.customAttributes||"string"!=typeof e||0===e.length)return void(0,h.R)(57);const p=(e,t)=>null==e?t:"number"==typeof e?e:e instanceof PerformanceMark?e.startTime:Number.NaN;if(f.start=p(d,0),f.end=p(u,c),Number.isNaN(f.start)||Number.isNaN(f.end))(0,h.R)(57);else{if(f.duration=f.end-f.start,!(f.duration<0))return(0,s.p)(a.Pl+a.V1,[f,e,i],void 0,t.K7.genericEvents,n.ee),f;(0,h.R)(58)}}function z(e,r={},n,i,c=(0,o.t)()){(0,s.p)(a.Pl+a.fF,[c,e,r,i],void 0,t.K7.genericEvents,n.ee)}var G=i(7508),Y=i(9566);const Z=["name","id","type"],q=new Map([[U,"addPageAction"],[F,"log"],[V,"measure"],[B,"noticeError"],[z,"recordCustomEvent"]]),X={experimental:(0,m.J)(()=>(0,h.R)(54,"newrelic.register")),disabled:(0,m.J)(()=>(0,h.R)(55)),invalidTarget:(0,m.J)(e=>(0,h.R)(48,e)),deregistered:(0,m.J)(()=>(0,h.R)(68))};function Q(e){(0,c.Y)(a.eY,function(t){return J(e,t)},e)}function J(e,r){X.experimental(),r||={},r.instance=(0,Y.LA)(8),r.type=H.fQ.MFE,r.licenseKey||=e.info.licenseKey,r.blocked=!1,("object"!=typeof r.tags||null===r.tags||Array.isArray(r.tags))&&(r.tags={}),r.parent??={get id(){return e.runtime.appMetadata.agents[0].entityGuid},type:H.fQ.BA};const n=(0,G.Qr)(),i={};Object.prototype.hasOwnProperty.call(r,"attributes")||Object.defineProperty(r,"attributes",{get:()=>({...i,"source.id":r.id,"source.name":r.name,"source.type":r.type,"parent.type":r.parent?.type||H.fQ.BA,"parent.id":r.parent?.id})}),Object.entries(r.tags).forEach(([e,t])=>{Z.includes(e)||(i["source.".concat(e)]=t)});let a=()=>{};const c=e.runtime.registeredEntities,d=e=>{r.blocked=!0,a=e};function u(e){return"string"==typeof e&&!!e.trim()&&e.trim().length<501}e.init.api.register.enabled||d(X.disabled),u(r.id)&&u(r.name)||d(()=>X.invalidTarget(r));const l={addPageAction:(t,n={})=>m(U,[t,{...i,...n},e],r),deregister:()=>{p(),d(X.deregistered)},log:(t,n={})=>m(F,[t,{...n,customAttributes:{...i,...n.customAttributes||{}}},e],r),measure:(t,n={})=>m(V,[t,{...n,customAttributes:{...i,...n.customAttributes||{}}},e],r),noticeError:(t,n={})=>m(B,[t,{...i,...n},e],r),recordCustomEvent:(t,n={})=>m(z,[t,{...i,...n},e],r),setApplicationVersion:e=>g("application.version",e),setCustomAttribute:(e,t)=>g(e,t),setUserId:e=>g("enduser.id",e),metadata:{get customAttributes(){return i},target:r,timings:n}},f=()=>(r.blocked&&a(),r.blocked);function p(){if(n.reportedAt)return;n.reportedAt=(0,o.t)();const e=n.fetchEnd-n.fetchStart,t=n.scriptEnd-n.scriptStart;l.recordCustomEvent("MicroFrontEndTiming",{assetUrl:n.asset,assetType:n.type,timeAlive:n.reportedAt-n.registeredAt,timeToBeRequested:n.fetchStart,timeToExecute:t,timeToFetch:e,timeToLoad:e+t,timeToRegister:n.registeredAt})}f()||(c.push(l),(0,x.G)(p));const g=(e,t)=>{f()||(i[e]=t)},m=(r,n,i)=>{if(f()&&r!==J)return;const a=(0,o.t)(),c=q.get(r)||"unknown";(0,s.p)(O.xV,["API/register/".concat(c,"/called")],void 0,t.K7.metrics,e.ee);try{return r(...n,i,a)}catch(e){(0,h.R)(50,e)}};return l}class ee extends y{static featureName=k.T;constructor(e){var t;super(e,k.T),t=e,(0,c.Y)(a.o5,(e,r)=>B(e,r,t),t),function(e){(0,c.Y)(a.bt,function(t){e.runtime.onerror=t},e)}(e),function(e){let t=0;(0,c.Y)(a.k6,function(e,r){++t>10||(this.runtime.releaseIds[e.slice(-200)]=(""+r).slice(-200))},e)}(e),Q(e);try{this.removeOnAbort=new AbortController}catch(e){}this.ee.on("internal-error",(t,r)=>{this.abortHandler&&(0,s.p)("ierr",[C(t),(0,o.t)(),!0,{},e.runtime.isRecording,r],void 0,this.featureName,this.ee)}),f.gm.addEventListener("unhandledrejection",t=>{this.abortHandler&&(0,s.p)("err",[L(t),(0,o.t)(),!1,{unhandledPromiseRejection:1},e.runtime.isRecording],void 0,this.featureName,this.ee)},(0,N.jT)(!1,this.removeOnAbort?.signal)),f.gm.addEventListener("error",t=>{this.abortHandler&&(0,s.p)("err",[I(t),(0,o.t)(),!1,{},e.runtime.isRecording],void 0,this.featureName,this.ee)},(0,N.jT)(!1,this.removeOnAbort?.signal)),this.abortHandler=this.#r,this.importAggregator(e,()=>i.e(478).then(i.bind(i,9377)))}#r(){this.removeOnAbort?.abort(),this.abortHandler=void 0}}var te=i(8990);let re=1;function ne(e){const t=typeof e;return!e||"object"!==t&&"function"!==t?-1:e===f.gm?0:(0,te.I)(e,"nr@id",function(){return re++})}function ie(e){if("string"==typeof e&&e.length)return e.length;if("object"==typeof e){if("undefined"!=typeof ArrayBuffer&&e instanceof ArrayBuffer&&e.byteLength)return e.byteLength;if("undefined"!=typeof Blob&&e instanceof Blob&&e.size)return e.size;if(!("undefined"!=typeof FormData&&e instanceof FormData))try{return(0,j.A)(e).length}catch(e){return}}}var se=i(8139),oe=i(7836),ae=i(3434);const ce={},de=["open","send"];function ue(e,t){var r=e||oe.ee;const n=function(e){return(e||oe.ee).get("xhr")}(r);if(void 0===f.gm.XMLHttpRequest)return n;if(ce[n.debugId]++)return n;ce[n.debugId]=1,(0,se.u)(r);var i=(0,ae.YM)(n),s=f.gm.XMLHttpRequest,o=f.gm.MutationObserver,a=f.gm.Promise,c=f.gm.setInterval,d="readystatechange",u=["onload","onerror","onabort","onloadstart","onloadend","onprogress","ontimeout"],l=[],p=f.gm.XMLHttpRequest=function(e){const r=new s(e),o=n.context(r);o.targets=(0,H.$5)(t);try{n.emit("new-xhr",[r],o),r.addEventListener(d,(a=o,function(){var e=this;e.readyState>3&&!a.resolved&&(a.resolved=!0,n.emit("xhr-resolved",[],e)),i.inPlace(e,u,"fn-",w)}),(0,N.jT)(!1))}catch(e){(0,h.R)(15,e);try{n.emit("internal-error",[e])}catch(e){}}var a;return r};function g(e,t){i.inPlace(t,["onreadystatechange"],"fn-",w)}if(function(e,t){for(var r in e)t[r]=e[r]}(s,p),p.prototype=s.prototype,i.inPlace(p.prototype,de,"-xhr-",w),n.on("send-xhr-start",function(e,t){g(e,t),function(e){l.push(e),o&&(m?m.then(b):c?c(b):(v=-v,y.data=v))}(t)}),n.on("open-xhr-start",g),o){var m=a&&a.resolve();if(!c&&!a){var v=1,y=document.createTextNode(v);new o(b).observe(y,{characterData:!0})}}else r.on("fn-end",function(e){e[0]&&e[0].type===d||b()});function b(){for(var e=0;e<l.length;e++)g(0,l[e]);l.length&&(l=[])}function w(e,t){return t}return n}var le="fetch-",fe=le+"body-",he=["arrayBuffer","blob","json","text","formData"],pe=f.gm.Request,ge=f.gm.Response,me="prototype";const ve={};function ye(e,t){const r=function(e){return(e||oe.ee).get("fetch")}(e);if(!(pe&&ge&&f.gm.fetch))return r;if(ve[r.debugId]++)return r;function n(e,n,i){var s=e[n];"function"==typeof s&&(e[n]=function(){var e=[...arguments];const n={},o=(0,H.$5)(t);var a;r.emit(i+"before-start",[e],n),n[oe.P]&&n[oe.P].dt&&(a=n[oe.P].dt);var c=s.apply(this,e);return r.emit(i+"start",[e,a],c),c.then(function(e){return r.emit(i+"end",[null,e,o],c),e},function(e){throw r.emit(i+"end",[e,void 0,o],c),e})})}return ve[r.debugId]=1,he.forEach(e=>{n(pe[me],e,fe),n(ge[me],e,fe)}),n(f.gm,"fetch",le),r.on(le+"end",function(e,t,n){var i=this;if(i.targets=n||[void 0],t){var s=t.headers.get("content-length");null!==s&&(i.rxSize=s),r.emit(le+"done",[null,t],i)}else r.emit(le+"done",[e],i)}),r}var be=i(7485);class we{constructor(e){this.agentRef=e}generateTracePayload(e){const t=this.agentRef.loader_config;if(!this.shouldGenerateTrace(e)||!t)return null;var r=(t.accountID||"").toString()||null,n=(t.agentID||"").toString()||null,i=(t.trustKey||"").toString()||null;if(!r||!n)return null;var s=(0,Y.ZF)(),o=(0,Y.el)(),a=Date.now(),c={spanId:s,traceId:o,timestamp:a};return(e.sameOrigin||this.isAllowedOrigin(e)&&this.useTraceContextHeadersForCors())&&(c.traceContextParentHeader=this.generateTraceContextParentHeader(s,o),c.traceContextStateHeader=this.generateTraceContextStateHeader(s,a,r,n,i)),(e.sameOrigin&&!this.excludeNewrelicHeader()||!e.sameOrigin&&this.isAllowedOrigin(e)&&this.useNewrelicHeaderForCors())&&(c.newrelicHeader=this.generateTraceHeader(s,o,a,r,n,i)),c}generateTraceContextParentHeader(e,t){return"00-"+t+"-"+e+"-01"}generateTraceContextStateHeader(e,t,r,n,i){return i+"@nr=0-1-"+r+"-"+n+"-"+e+"----"+t}generateTraceHeader(e,t,r,n,i,s){if(!("function"==typeof f.gm?.btoa))return null;var o={v:[0,1],d:{ty:"Browser",ac:n,ap:i,id:e,tr:t,ti:r}};return s&&n!==s&&(o.d.tk=s),btoa((0,j.A)(o))}shouldGenerateTrace(e){return this.agentRef.init?.distributed_tracing?.enabled&&this.isAllowedOrigin(e)}isAllowedOrigin(e){var t=!1;const r=this.agentRef.init?.distributed_tracing;if(e.sameOrigin)t=!0;else if(r?.allowed_origins instanceof Array)for(var n=0;n<r.allowed_origins.length;n++){var i=(0,be.D)(r.allowed_origins[n]);if(e.hostname===i.hostname&&e.protocol===i.protocol&&e.port===i.port){t=!0;break}}return t}excludeNewrelicHeader(){var e=this.agentRef.init?.distributed_tracing;return!!e&&!!e.exclude_newrelic_header}useNewrelicHeaderForCors(){var e=this.agentRef.init?.distributed_tracing;return!!e&&!1!==e.cors_use_newrelic_header}useTraceContextHeadersForCors(){var e=this.agentRef.init?.distributed_tracing;return!!e&&!!e.cors_use_tracecontext_headers}}var Re=i(9300),Ee=i(7295);function Te(e){return"string"==typeof e?e:e instanceof(0,T.dV)().o.REQ?e.url:f.gm?.URL&&e instanceof URL?e.href:void 0}var Ae=["load","error","abort","timeout"],xe=Ae.length,Se=(0,T.dV)().o.REQ,_e=(0,T.dV)().o.XHR;const Oe="X-NewRelic-App-Data";class Pe extends y{static featureName=Re.T;constructor(e){super(e,Re.T),this.dt=new we(e),this.handler=(e,t,r,n)=>(0,s.p)(e,t,r,n,this.ee);try{const e={xmlhttprequest:"xhr",fetch:"fetch",beacon:"beacon"};f.gm?.performance?.getEntriesByType("resource").forEach(r=>{if(r.initiatorType in e&&0!==r.responseStatus){const n={status:r.responseStatus},i={rxSize:r.transferSize,duration:Math.floor(r.duration),cbTime:0};ke(n,r.name),this.handler("xhr",[n,i,r.startTime,r.responseEnd,e[r.initiatorType]],void 0,t.K7.ajax)}})}catch(e){}ye(this.ee,e),ue(this.ee,e),function(e,r,n,i){function a(e){var t=this;t.totalCbs=0,t.called=0,t.cbTime=0,t.end=T,t.ended=!1,t.xhrGuids={},t.lastSize=null,t.loadCaptureCalled=!1,t.params=this.params||{},t.metrics=this.metrics||{},t.latestLongtaskEnd=0,e.addEventListener("load",function(r){x(t,e)},(0,N.jT)(!1)),f.lR||e.addEventListener("progress",function(e){t.lastSize=e.loaded},(0,N.jT)(!1))}function c(e){this.params={method:e[0]},ke(this,e[1]),this.metrics={}}function d(t,r){e.loader_config.xpid&&this.sameOrigin&&r.setRequestHeader("X-NewRelic-ID",e.loader_config.xpid);var n=i.generateTracePayload(this.parsedOrigin);if(n){var s=!1;n.newrelicHeader&&(r.setRequestHeader("newrelic",n.newrelicHeader),s=!0),n.traceContextParentHeader&&(r.setRequestHeader("traceparent",n.traceContextParentHeader),n.traceContextStateHeader&&r.setRequestHeader("tracestate",n.traceContextStateHeader),s=!0),s&&(this.dt=n)}}function u(e,t){var n=this.metrics,i=e[0],s=this;if(n&&i){var a=ie(i);a&&(n.txSize=a)}this.startTime=(0,o.t)(),this.body=i,this.listener=function(e){try{"abort"!==e.type||s.loadCaptureCalled||(s.params.aborted=!0),("load"!==e.type||s.called===s.totalCbs&&(s.onloadCalled||"function"!=typeof t.onload)&&"function"==typeof s.end)&&s.end(t)}catch(e){try{r.emit("internal-error",[e])}catch(e){}}};for(var c=0;c<xe;c++)t.addEventListener(Ae[c],this.listener,(0,N.jT)(!1))}function l(e,t,r){this.cbTime+=e,t?this.onloadCalled=!0:this.called+=1,this.called!==this.totalCbs||!this.onloadCalled&&"function"==typeof r.onload||"function"!=typeof this.end||this.end(r)}function h(e,t){var r=""+ne(e)+!!t;this.xhrGuids&&!this.xhrGuids[r]&&(this.xhrGuids[r]=!0,this.totalCbs+=1)}function p(e,t){var r=""+ne(e)+!!t;this.xhrGuids&&this.xhrGuids[r]&&(delete this.xhrGuids[r],this.totalCbs-=1)}function g(){this.endTime=(0,o.t)()}function m(e,t){t instanceof _e&&"load"===e[0]&&r.emit("xhr-load-added",[e[1],e[2]],t)}function v(e,t){t instanceof _e&&"load"===e[0]&&r.emit("xhr-load-removed",[e[1],e[2]],t)}function y(e,t,r){t instanceof _e&&("onload"===r&&(this.onload=!0),("load"===(e[0]&&e[0].type)||this.onload)&&(this.xhrCbStart=(0,o.t)()))}function b(e,t){this.xhrCbStart&&r.emit("xhr-cb-time",[(0,o.t)()-this.xhrCbStart,this.onload,t],t)}function w(e){var t,r=e[1]||{};if("string"==typeof e[0]?0===(t=e[0]).length&&f.RI&&(t=""+f.gm.location.href):e[0]&&e[0].url?t=e[0].url:f.gm?.URL&&e[0]&&e[0]instanceof URL?t=e[0].href:"function"==typeof e[0].toString&&(t=e[0].toString()),"string"==typeof t&&0!==t.length){t&&(this.parsedOrigin=(0,be.D)(t),this.sameOrigin=this.parsedOrigin.sameOrigin);var n=i.generateTracePayload(this.parsedOrigin);if(n&&(n.newrelicHeader||n.traceContextParentHeader))if(e[0]&&e[0].headers)a(e[0].headers,n)&&(this.dt=n);else{var s={};for(var o in r)s[o]=r[o];s.headers=new Headers(r.headers||{}),a(s.headers,n)&&(this.dt=n),e.length>1?e[1]=s:e.push(s)}}function a(e,t){var r=!1;return t.newrelicHeader&&(e.set("newrelic",t.newrelicHeader),r=!0),t.traceContextParentHeader&&(e.set("traceparent",t.traceContextParentHeader),t.traceContextStateHeader&&e.set("tracestate",t.traceContextStateHeader),r=!0),r}}function R(e,t){this.params={},this.metrics={},this.startTime=(0,o.t)(),this.dt=t;let[r,n={}]=e;ke(this,Te(r));const i=(""+(r&&r instanceof Se&&r.method||n.method||"GET")).toUpperCase();this.params.method=i,this.body=n.body,this.txSize=ie(n.body)||0}function E(e,t){if(this.endTime=(0,o.t)(),this.params||(this.params={}),(0,Ee.iW)(this.params))return;let r;this.params.status=t?t.status:0,"string"==typeof this.rxSize&&this.rxSize.length>0&&(r=+this.rxSize);const n={txSize:this.txSize,rxSize:r,duration:(0,o.t)()-this.startTime},i=[this.params,n,this.startTime,this.endTime,"fetch"];this.targets.forEach(e=>A(i,this,e))}function T(e){const t=this.params,r=this.metrics;if(this.ended)return;this.ended=!0;for(let t=0;t<xe;t++)e.removeEventListener(Ae[t],this.listener,!1);if(t.aborted)return;if((0,Ee.iW)(t))return;r.duration=(0,o.t)()-this.startTime,this.loadCaptureCalled||4!==e.readyState?null==t.status&&(t.status=0):x(this,e),r.cbTime=this.cbTime;const n=[t,r,this.startTime,this.endTime,"xhr"];this.targets.forEach(e=>A(n,this,e))}function A(e,r,i){n("xhr",[...e,i],r,t.K7.ajax)}function x(e,n){e.params.status=n.status;var i=function(e,t){var r=e.responseType;return"json"===r&&null!==t?t:"arraybuffer"===r||"blob"===r||"json"===r?ie(e.response):"text"===r||""===r||void 0===r?ie(e.responseText):void 0}(n,e.lastSize);if(i&&(e.metrics.rxSize=i),e.sameOrigin&&n.getAllResponseHeaders().indexOf(Oe)>=0){var o=n.getResponseHeader(Oe);o&&((0,s.p)(O.rs,["Ajax/CrossApplicationTracing/Header/Seen"],void 0,t.K7.metrics,r),e.params.cat=o.split(", ").pop())}e.loadCaptureCalled=!0}r.on("new-xhr",a),r.on("open-xhr-start",c),r.on("open-xhr-end",d),r.on("send-xhr-start",u),r.on("xhr-cb-time",l),r.on("xhr-load-added",h),r.on("xhr-load-removed",p),r.on("xhr-resolved",g),r.on("addEventListener-end",m),r.on("removeEventListener-end",v),r.on("fn-end",b),r.on("fetch-before-start",w),r.on("fetch-start",R),r.on("fn-start",y),r.on("fetch-done",E)}(e,this.ee,this.handler,this.dt),this.importAggregator(e,()=>i.e(478).then(i.bind(i,3845)))}}function ke(e,t){var r=(0,be.D)(t),n=e.params||e;n.hostname=r.hostname,n.port=r.port,n.protocol=r.protocol,n.host=r.hostname+":"+r.port,n.pathname=r.pathname,e.parsedOrigin=r,e.sameOrigin=r.sameOrigin}const Ne={},je=["pushState","replaceState"];function De(e){const t=function(e){return(e||oe.ee).get("history")}(e);return!f.RI||Ne[t.debugId]++||(Ne[t.debugId]=1,(0,ae.YM)(t).inPlace(window.history,je,"-")),t}var Ce=i(3738);function Le(e){(0,c.Y)(a.BL,function(r=Date.now()){const n=r-f.WN;n<0&&(0,h.R)(62,r),(0,s.p)(O.XG,[a.BL,{time:n}],void 0,t.K7.metrics,e.ee),e.addToTrace({name:a.BL,start:r,origin:"nr"}),(0,s.p)(a.Pl+a.hG,[n,a.BL],void 0,t.K7.genericEvents,e.ee)},e)}const{He:Ie,bD:Me,d3:Be,Kp:He,TZ:Ke,Lc:We,uP:Fe,Rz:Ue}=Ce;class Ve extends y{static featureName=Ke;constructor(e){var r;super(e,Ke),r=e,(0,c.Y)(a.U2,function(e){if(!(e&&"object"==typeof e&&e.name&&e.start))return;const n={n:e.name,s:e.start-f.WN,e:(e.end||e.start)-f.WN,o:e.origin||"",t:"api"};n.s<0||n.e<0||n.e<n.s?(0,h.R)(61,{start:n.s,end:n.e}):(0,s.p)("bstApi",[n],void 0,t.K7.sessionTrace,r.ee)},r),Le(e);if(!(0,g.V)(e.init))return void this.deregisterDrain();const n=this.ee;let d;De(n),this.eventsEE=(0,se.u)(n),this.eventsEE.on(Fe,function(e,t){this.bstStart=(0,o.t)()}),this.eventsEE.on(We,function(e,r){(0,s.p)("bst",[e[0],r,this.bstStart,(0,o.t)()],void 0,t.K7.sessionTrace,n)}),n.on(Ue+Be,function(e){this.time=(0,o.t)(),this.startPath=location.pathname+location.hash}),n.on(Ue+He,function(e){(0,s.p)("bstHist",[location.pathname+location.hash,this.startPath,this.time],void 0,t.K7.sessionTrace,n)});try{d=new PerformanceObserver(e=>{const r=e.getEntries();(0,s.p)(Ie,[r],void 0,t.K7.sessionTrace,n)}),d.observe({type:Me,buffered:!0})}catch(e){}this.importAggregator(e,()=>i.e(478).then(i.bind(i,6974)),{resourceObserver:d})}}var ze=i(6344);class Ge extends y{static featureName=ze.TZ;#n;recorder;constructor(e){var r;let n;super(e,ze.TZ),r=e,(0,c.Y)(a.CH,function(){(0,s.p)(a.CH,[],void 0,t.K7.sessionReplay,r.ee)},r),function(e){(0,c.Y)(a.Tb,function(){(0,s.p)(a.Tb,[],void 0,t.K7.sessionReplay,e.ee)},e)}(e);try{n=JSON.parse(localStorage.getItem("".concat(w.H3,"_").concat(w.uh)))}catch(e){}(0,p.SR)(e.init)&&this.ee.on(a.CH,()=>this.#i()),this.#s(n)&&this.importRecorder().then(e=>{e.startRecording(ze.Qb.PRELOAD,n?.sessionReplayMode)}),this.importAggregator(this.agentRef,()=>i.e(478).then(i.bind(i,6167)),this),this.ee.on("err",e=>{this.blocked||this.agentRef.runtime.isRecording&&(this.errorNoticed=!0,(0,s.p)(ze.Vh,[e],void 0,this.featureName,this.ee))})}#s(e){return e&&(e.sessionReplayMode===w.g.FULL||e.sessionReplayMode===w.g.ERROR)||(0,p.Aw)(this.agentRef.init)}importRecorder(){return this.recorder?Promise.resolve(this.recorder):(this.#n??=Promise.all([i.e(478),i.e(249)]).then(i.bind(i,4866)).then(({Recorder:e})=>(this.recorder=new e(this),this.recorder)).catch(e=>{throw this.ee.emit("internal-error",[e]),this.blocked=!0,e}),this.#n)}#i(){this.blocked||(this.featAggregate?this.featAggregate.mode!==w.g.FULL&&this.featAggregate.initializeRecording(w.g.FULL,!0,ze.Qb.API):this.importRecorder().then(()=>{this.recorder.startRecording(ze.Qb.API,w.g.FULL)}))}}var Ye=i(3962);class Ze extends y{static featureName=Ye.TZ;constructor(e){if(super(e,Ye.TZ),function(e){const r=e.ee.get("tracer");function n(){}(0,c.Y)(a.dT,function(e){return(new n).get("object"==typeof e?e:{})},e);const i=n.prototype={createTracer:function(n,i){var a={},c=this,d="function"==typeof i;return(0,s.p)(O.xV,["API/createTracer/called"],void 0,t.K7.metrics,e.ee),function(){if(r.emit((d?"":"no-")+"fn-start",[(0,o.t)(),c,d],a),d)try{return i.apply(this,arguments)}catch(e){const t="string"==typeof e?new Error(e):e;throw r.emit("fn-err",[arguments,this,t],a),t}finally{r.emit("fn-end",[(0,o.t)()],a)}}}};["actionText","setName","setAttribute","save","ignore","onEnd","getContext","end","get"].forEach(r=>{c.Y.apply(this,[r,function(){return(0,s.p)(a.hw+r,[performance.now(),...arguments],this,t.K7.softNav,e.ee),this},e,i])}),(0,c.Y)(a.PA,function(){(0,s.p)(a.hw+"routeName",[performance.now(),...arguments],void 0,t.K7.softNav,e.ee)},e)}(e),!f.RI||!(0,T.dV)().o.MO)return;const r=De(this.ee);try{this.removeOnAbort=new AbortController}catch(e){}Ye.tC.forEach(e=>{(0,N.sp)(e,e=>{l(e)},!0,this.removeOnAbort?.signal)});const n=()=>(0,s.p)("newURL",[(0,o.t)(),""+window.location],void 0,this.featureName,this.ee);r.on("pushState-end",n),r.on("replaceState-end",n),(0,N.sp)(Ye.OV,e=>{l(e),(0,s.p)("newURL",[e.timeStamp,""+window.location],void 0,this.featureName,this.ee)},!0,this.removeOnAbort?.signal);let d=!1;const u=new((0,T.dV)().o.MO)((e,t)=>{d||(d=!0,requestAnimationFrame(()=>{(0,s.p)("newDom",[(0,o.t)()],void 0,this.featureName,this.ee),d=!1}))}),l=(0,m.s)(e=>{"loading"!==document.readyState&&((0,s.p)("newUIEvent",[e],void 0,this.featureName,this.ee),u.observe(document.body,{attributes:!0,childList:!0,subtree:!0,characterData:!0}))},100,{leading:!0});this.abortHandler=function(){this.removeOnAbort?.abort(),u.disconnect(),this.abortHandler=void 0},this.importAggregator(e,()=>i.e(478).then(i.bind(i,4393)),{domObserver:u})}}var qe=i(3333),Xe=i(9119);const $e={},Qe=new Set;function Je(e){return"string"==typeof e?{type:"string",size:(new TextEncoder).encode(e).length}:e instanceof ArrayBuffer?{type:"ArrayBuffer",size:e.byteLength}:e instanceof Blob?{type:"Blob",size:e.size}:e instanceof DataView?{type:"DataView",size:e.byteLength}:ArrayBuffer.isView(e)?{type:"TypedArray",size:e.byteLength}:{type:"unknown",size:0}}class et{constructor(e,t){this.timestamp=(0,o.t)(),this.currentUrl=(0,Xe.L)(window.location.href),this.socketId=(0,Y.LA)(8),this.requestedUrl=(0,Xe.L)(e),this.requestedProtocols=Array.isArray(t)?t.join(","):t||"",this.openedAt=void 0,this.protocol=void 0,this.extensions=void 0,this.binaryType=void 0,this.messageOrigin=void 0,this.messageCount=0,this.messageBytes=0,this.messageBytesMin=0,this.messageBytesMax=0,this.messageTypes=void 0,this.sendCount=0,this.sendBytes=0,this.sendBytesMin=0,this.sendBytesMax=0,this.sendTypes=void 0,this.closedAt=void 0,this.closeCode=void 0,this.closeReason="unknown",this.closeWasClean=void 0,this.connectedDuration=0,this.hasErrors=void 0}}class tt extends y{static featureName=qe.TZ;constructor(e){super(e,qe.TZ);const r=e.init.feature_flags.includes("websockets"),n=!e.init.feature_flags.includes("no_spv"),d=[e.init.page_action.enabled,e.init.performance.capture_marks,e.init.performance.capture_measures,e.init.performance.resources.enabled,e.init.user_actions.enabled,r,n];var u;let l;if(u=e,(0,c.Y)(a.hG,(e,t)=>U(e,t,u),u),function(e){(0,c.Y)(a.fF,(t,r)=>z(t,r,e),e)}(e),Le(e),Q(e),function(e){(0,c.Y)(a.V1,(t,r)=>V(t,r,e),e)}(e),this.removeOnAbort=new AbortController,this.abortHandler=()=>{this.removeOnAbort.abort(),this.abortHandler=void 0},r){const h=function(e){if(!(0,T.dV)().o.WS)return e;const t=e.get("websockets");if($e[t.debugId]++)return t;$e[t.debugId]=1,(0,x.G)(()=>{const e=(0,o.t)();Qe.forEach(r=>{r.nrData.closedAt=e,r.nrData.closeCode=1001,r.nrData.closeReason="Page navigating away",r.nrData.closeWasClean=!1,r.nrData.openedAt&&(r.nrData.connectedDuration=e-r.nrData.openedAt),t.emit("ws",[r.nrData],r)})});class r extends WebSocket{static name="WebSocket";static toString(){return"function WebSocket() { [native code] }"}toString(){return"[object WebSocket]"}get[Symbol.toStringTag](){return r.name}#o(e){(e.__newrelic??={}).socketId=this.nrData.socketId,this.nrData.hasErrors??=!0}constructor(...e){super(...e),this.nrData=new et(e[0],e[1]),this.addEventListener("open",()=>{this.nrData.openedAt=(0,o.t)(),["protocol","extensions","binaryType"].forEach(e=>{this.nrData[e]=this[e]}),Qe.add(this)}),this.addEventListener("message",e=>{const{type:t,size:r}=Je(e.data);this.nrData.messageOrigin??=(0,Xe.L)(e.origin),this.nrData.messageCount++,this.nrData.messageBytes+=r,this.nrData.messageBytesMin=Math.min(this.nrData.messageBytesMin||1/0,r),this.nrData.messageBytesMax=Math.max(this.nrData.messageBytesMax,r),(this.nrData.messageTypes??"").includes(t)||(this.nrData.messageTypes=this.nrData.messageTypes?"".concat(this.nrData.messageTypes,",").concat(t):t)}),this.addEventListener("close",e=>{this.nrData.closedAt=(0,o.t)(),this.nrData.closeCode=e.code,e.reason&&(this.nrData.closeReason=e.reason),this.nrData.closeWasClean=e.wasClean,this.nrData.connectedDuration=this.nrData.closedAt-this.nrData.openedAt,Qe.delete(this),t.emit("ws",[this.nrData],this)})}addEventListener(e,t,...r){const n=this,i="function"==typeof t?function(...e){try{return t.apply(this,e)}catch(e){throw n.#o(e),e}}:t?.handleEvent?{handleEvent:function(...e){try{return t.handleEvent.apply(t,e)}catch(e){throw n.#o(e),e}}}:t;return super.addEventListener(e,i,...r)}send(e){if(this.readyState===WebSocket.OPEN){const{type:t,size:r}=Je(e);this.nrData.sendCount++,this.nrData.sendBytes+=r,this.nrData.sendBytesMin=Math.min(this.nrData.sendBytesMin||1/0,r),this.nrData.sendBytesMax=Math.max(this.nrData.sendBytesMax,r),(this.nrData.sendTypes??"").includes(t)||(this.nrData.sendTypes=this.nrData.sendTypes?"".concat(this.nrData.sendTypes,",").concat(t):t)}try{return super.send(e)}catch(e){throw this.#o(e),e}}close(...e){try{super.close(...e)}catch(e){throw this.#o(e),e}}}return f.gm.WebSocket=r,t}(this.ee);h.on("ws",e=>{(0,s.p)("ws-complete",[e],void 0,this.featureName,this.ee)})}if(n&&f.gm.addEventListener("securitypolicyviolation",e=>{(0,s.p)("spv",[e],void 0,t.K7.genericEvents,this.ee)},(0,N.jT)(!1,this.removeOnAbort.signal)),f.RI){if(ye(this.ee,e),ue(this.ee,e),l=De(this.ee),e.init.user_actions.enabled){function p(t){const r=(0,be.D)(t);return e.beacons.includes(r.hostname+":"+r.port)}function g(){l.emit("navChange")}qe.Zp.forEach(e=>(0,N.sp)(e,e=>(0,s.p)("ua",[e],void 0,this.featureName,this.ee),!0)),qe.qN.forEach(e=>{const t=(0,m.s)(e=>{(0,s.p)("ua",[e],void 0,this.featureName,this.ee)},500,{leading:!0});(0,N.sp)(e,t)}),f.gm.addEventListener("error",()=>{(0,s.p)("uaErr",[],void 0,t.K7.genericEvents,this.ee)},(0,N.jT)(!1,this.removeOnAbort.signal)),this.ee.on("open-xhr-start",(e,r)=>{p(e[1])||r.addEventListener("readystatechange",()=>{2===r.readyState&&(0,s.p)("uaXhr",[],void 0,t.K7.genericEvents,this.ee)},(0,N.jT)(void 0,this.removeOnAbort.signal))}),this.ee.on("fetch-start",e=>{e.length>=1&&!p(Te(e[0]))&&(0,s.p)("uaXhr",[],void 0,t.K7.genericEvents,this.ee)}),l.on("pushState-end",g),l.on("replaceState-end",g),window.addEventListener("hashchange",g,(0,N.jT)(!0,this.removeOnAbort.signal)),window.addEventListener("popstate",g,(0,N.jT)(!0,this.removeOnAbort.signal))}if(e.init.performance.resources.enabled&&f.gm.PerformanceObserver?.supportedEntryTypes.includes("resource")){new PerformanceObserver(e=>{e.getEntries().forEach(e=>{(0,s.p)("browserPerformance.resource",[e],void 0,this.featureName,this.ee)})}).observe({type:"resource",buffered:!0})}}d.some(e=>e)?this.importAggregator(e,()=>i.e(478).then(i.bind(i,8019))):this.deregisterDrain()}}var rt=i(2646);const nt=new Map;function it(e,t,r,n,i=!0,s){if("object"!=typeof t||!t||"string"!=typeof r||!r||"function"!=typeof t[r])return(0,h.R)(29);const o=function(e){return(e||oe.ee).get("logger")}(e),a=(0,ae.YM)(o,void 0,s),c=new rt.y(oe.P);c.level=n.level,c.customAttributes=n.customAttributes,c.autoCaptured=i;const d=t[r]?.[ae.Jt]||t[r];return nt.set(d,c),a.inPlace(t,[r],"wrap-logger-",()=>nt.get(d),void 0,!0),o}var st=i(1910);class ot extends y{static featureName=K.TZ;constructor(e){var t;super(e,K.TZ),t=e,(0,c.Y)(a.$9,(e,r)=>F(e,r,t),t),function(e){(0,c.Y)(a.Wb,(t,r,{customAttributes:n={},level:i=K.p_.INFO}={})=>{it(e.ee,t,r,{customAttributes:n,level:i},!1,e)},e)}(e),Q(e);const r=this.ee;["log","error","warn","info","debug","trace"].forEach(t=>{(0,st.i)(f.gm.console[t]),it(r,f.gm.console,t,{level:"log"===t?"info":t},void 0,e)}),this.ee.on("wrap-logger-end",function([e],t,n,i=[]){const{level:s,customAttributes:o,autoCaptured:a}=this;i.forEach(t=>{(0,W.R)(r,e,o,s,a,t)})}),this.importAggregator(e,()=>i.e(478).then(i.bind(i,5288)))}}new A({features:[Pe,E,_,Ve,Ge,P,ee,tt,ot,Ze],loaderType:"spa"})})()})();</script>
    <meta name="viewport" content="width=device-width" />
    <link rel="author" href="https://today.usc.edu/wp-content/themes/usc-communications-2023/humans.txt" />
    <link rel="pingback" href="https://today.usc.edu/xmlrpc.php" />
    <link rel="profile" href="http://gmpg.org/xfn/11">
    <link rel="apple-touch-icon" sizes="180x180" href="/wp-content/themes/usc-communications-2023/favicon/apple-touch-icon.png">
    <link rel="icon" type="image/png" sizes="32x32" href="/wp-content/themes/usc-communications-2023/favicon/favicon-32x32.png">
    <link rel="icon" type="image/png" sizes="16x16" href="/wp-content/themes/usc-communications-2023/favicon/favicon-16x16.png">
    <link rel="manifest" href="/wp-content/themes/usc-communications-2023/favicon/site.webmanifest">
    <link rel="mask-icon" href="/wp-content/themes/usc-communications-2023/favicon/safari-pinned-tab.svg" color="#990000">
    <link rel="shortcut icon" href="/wp-content/themes/usc-communications-2023/favicon/favicon.ico">
    <meta name="msapplication-TileColor" content="#da532c">
    <meta name="msapplication-config" content="/wp-content/themes/usc-communications-2023/favicon/browserconfig.xml">
    <meta name="theme-color" content="#ffffff">
    <meta name='robots' content='index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1' />
<script id="cookieyes" type="text/javascript" src="https://cdn-cookieyes.com/client_data/6ba326e6bf5a7fbc6058334e/script.js"></script>
	<!-- This site is optimized with the Yoast SEO Premium plugin v27.3 (Yoast SEO v27.3) - https://yoast.com/product/yoast-seo-premium-wordpress/ -->
	<title>USC News - USC Today</title>
	<meta name="description" content="Keep up to date with the latest news and updates from USC. Get the inside scoop on university news, research and initiatives." />
	<link rel="canonical" href="https://today.usc.edu/news/" />
	<meta property="og:locale" content="en_US" />
	<meta property="og:type" content="article" />
	<meta property="og:title" content="USC News" />
	<meta property="og:description" content="Keep up to date with the latest news and updates from USC. Get the inside scoop on university news, research and initiatives." />
	<meta property="og:url" content="https://today.usc.edu/news/" />
	<meta property="og:site_name" content="USC Today" />
	<meta property="article:publisher" content="https://www.facebook.com/usc/" />
	<meta property="article:modified_time" content="2026-05-07T15:36:54+00:00" />
	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:site" content="@usc" />
	<script type="application/ld+json" class="yoast-schema-graph">{"@context":"https:\/\/schema.org","@graph":[{"@type":"WebPage","@id":"https:\/\/today.usc.edu\/news\/","url":"https:\/\/today.usc.edu\/news\/","name":"USC News - USC Today","isPartOf":{"@id":"https:\/\/today.usc.edu\/#website"},"datePublished":"2023-06-29T11:20:58+00:00","dateModified":"2026-05-07T15:36:54+00:00","description":"Keep up to date with the latest news and updates from USC. Get the inside scoop on university news, research and initiatives.","breadcrumb":{"@id":"https:\/\/today.usc.edu\/news\/#breadcrumb"},"inLanguage":"en-US","potentialAction":[{"@type":"ReadAction","target":["https:\/\/today.usc.edu\/news\/"]}]},{"@type":"BreadcrumbList","@id":"https:\/\/today.usc.edu\/news\/#breadcrumb","itemListElement":[{"@type":"ListItem","position":1,"name":"Home","item":"https:\/\/today.usc.edu\/"},{"@type":"ListItem","position":2,"name":"USC News"}]},{"@type":"WebSite","@id":"https:\/\/today.usc.edu\/#website","url":"https:\/\/today.usc.edu\/","name":"USC Today","description":"University of Southern California News","publisher":{"@id":"https:\/\/today.usc.edu\/#organization"},"potentialAction":[{"@type":"SearchAction","target":{"@type":"EntryPoint","urlTemplate":"https:\/\/today.usc.edu\/?s={search_term_string}"},"query-input":{"@type":"PropertyValueSpecification","valueRequired":true,"valueName":"search_term_string"}}],"inLanguage":"en-US"},{"@type":"Organization","@id":"https:\/\/today.usc.edu\/#organization","name":"University of Southern California","url":"https:\/\/today.usc.edu\/","logo":{"@type":"ImageObject","inLanguage":"en-US","@id":"https:\/\/today.usc.edu\/#\/schema\/logo\/image\/","url":"https:\/\/today.usc.edu\/wp-content\/uploads\/2023\/10\/PrimShield_Mono_RegUse_Card-Blk_RGB.png","contentUrl":"https:\/\/today.usc.edu\/wp-content\/uploads\/2023\/10\/PrimShield_Mono_RegUse_Card-Blk_RGB.png","width":796,"height":391,"caption":"University of Southern California"},"image":{"@id":"https:\/\/today.usc.edu\/#\/schema\/logo\/image\/"},"sameAs":["https:\/\/www.facebook.com\/usc\/","https:\/\/x.com\/usc","https:\/\/www.instagram.com\/uscedu\/","https:\/\/www.linkedin.com\/school\/3084","https:\/\/www.youtube.com\/user\/USC"]}]}</script>
	<!-- / Yoast SEO Premium plugin. -->


<link rel='dns-prefetch' href='//www.googletagmanager.com' />
<link rel='dns-prefetch' href='//use.typekit.net' />
<link rel='dns-prefetch' href='//fonts.googleapis.com' />
<link rel="alternate" type="application/rss+xml" title="USC Today &raquo; Feed" href="https://today.usc.edu/feed/" />
<link rel="alternate" type="application/rss+xml" title="USC Today &raquo; Comments Feed" href="https://today.usc.edu/comments/feed/" />
<link rel="alternate" title="oEmbed (JSON)" type="application/json+oembed" href="https://today.usc.edu/wp-json/oembed/1.0/embed?url=https%3A%2F%2Ftoday.usc.edu%2Fnews%2F" />
<link rel="alternate" title="oEmbed (XML)" type="text/xml+oembed" href="https://today.usc.edu/wp-json/oembed/1.0/embed?url=https%3A%2F%2Ftoday.usc.edu%2Fnews%2F&#038;format=xml" />
<link rel='stylesheet' id='wpra-lightbox-css' href='https://today.usc.edu/wp-content/plugins/wp-rss-aggregator/core/css/jquery-colorbox.css?ver=1.4.33' type='text/css' media='all' />
<style id='wp-emoji-styles-inline-css' type='text/css'>

	img.wp-smiley, img.emoji {
		display: inline !important;
		border: none !important;
		box-shadow: none !important;
		height: 1em !important;
		width: 1em !important;
		margin: 0 0.07em !important;
		vertical-align: -0.1em !important;
		background: none !important;
		padding: 0 !important;
	}
/*# sourceURL=wp-emoji-styles-inline-css */
</style>
<style id='wp-block-library-inline-css' type='text/css'>
:root{--wp-block-synced-color:#7a00df;--wp-block-synced-color--rgb:122,0,223;--wp-bound-block-color:var(--wp-block-synced-color);--wp-editor-canvas-background:#ddd;--wp-admin-theme-color:#007cba;--wp-admin-theme-color--rgb:0,124,186;--wp-admin-theme-color-darker-10:#006ba1;--wp-admin-theme-color-darker-10--rgb:0,107,160.5;--wp-admin-theme-color-darker-20:#005a87;--wp-admin-theme-color-darker-20--rgb:0,90,135;--wp-admin-border-width-focus:2px}@media (min-resolution:192dpi){:root{--wp-admin-border-width-focus:1.5px}}.wp-element-button{cursor:pointer}:root .has-very-light-gray-background-color{background-color:#eee}:root .has-very-dark-gray-background-color{background-color:#313131}:root .has-very-light-gray-color{color:#eee}:root .has-very-dark-gray-color{color:#313131}:root .has-vivid-green-cyan-to-vivid-cyan-blue-gradient-background{background:linear-gradient(135deg,#00d084,#0693e3)}:root .has-purple-crush-gradient-background{background:linear-gradient(135deg,#34e2e4,#4721fb 50%,#ab1dfe)}:root .has-hazy-dawn-gradient-background{background:linear-gradient(135deg,#faaca8,#dad0ec)}:root .has-subdued-olive-gradient-background{background:linear-gradient(135deg,#fafae1,#67a671)}:root .has-atomic-cream-gradient-background{background:linear-gradient(135deg,#fdd79a,#004a59)}:root .has-nightshade-gradient-background{background:linear-gradient(135deg,#330968,#31cdcf)}:root .has-midnight-gradient-background{background:linear-gradient(135deg,#020381,#2874fc)}:root{--wp--preset--font-size--normal:16px;--wp--preset--font-size--huge:42px}.has-regular-font-size{font-size:1em}.has-larger-font-size{font-size:2.625em}.has-normal-font-size{font-size:var(--wp--preset--font-size--normal)}.has-huge-font-size{font-size:var(--wp--preset--font-size--huge)}.has-text-align-center{text-align:center}.has-text-align-left{text-align:left}.has-text-align-right{text-align:right}.has-fit-text{white-space:nowrap!important}#end-resizable-editor-section{display:none}.aligncenter{clear:both}.items-justified-left{justify-content:flex-start}.items-justified-center{justify-content:center}.items-justified-right{justify-content:flex-end}.items-justified-space-between{justify-content:space-between}.screen-reader-text{border:0;clip-path:inset(50%);height:1px;margin:-1px;overflow:hidden;padding:0;position:absolute;width:1px;word-wrap:normal!important}.screen-reader-text:focus{background-color:#ddd;clip-path:none;color:#444;display:block;font-size:1em;height:auto;left:5px;line-height:normal;padding:15px 23px 14px;text-decoration:none;top:5px;width:auto;z-index:100000}html :where(.has-border-color){border-style:solid}html :where([style*=border-top-color]){border-top-style:solid}html :where([style*=border-right-color]){border-right-style:solid}html :where([style*=border-bottom-color]){border-bottom-style:solid}html :where([style*=border-left-color]){border-left-style:solid}html :where([style*=border-width]){border-style:solid}html :where([style*=border-top-width]){border-top-style:solid}html :where([style*=border-right-width]){border-right-style:solid}html :where([style*=border-bottom-width]){border-bottom-style:solid}html :where([style*=border-left-width]){border-left-style:solid}html :where(img[class*=wp-image-]){height:auto;max-width:100%}:where(figure){margin:0 0 1em}html :where(.is-position-sticky){--wp-admin--admin-bar--position-offset:var(--wp-admin--admin-bar--height,0px)}@media screen and (max-width:600px){html :where(.is-position-sticky){--wp-admin--admin-bar--position-offset:0px}}

/*# sourceURL=wp-block-library-inline-css */
</style>
<style id='classic-theme-styles-inline-css' type='text/css'>
/*! This file is auto-generated */
.wp-block-button__link{color:#fff;background-color:#32373c;border-radius:9999px;box-shadow:none;text-decoration:none;padding:calc(.667em + 2px) calc(1.333em + 2px);font-size:1.125em}.wp-block-file__button{background:#32373c;color:#fff;text-decoration:none}
/*# sourceURL=/wp-includes/css/classic-themes.min.css */
</style>
<style id='global-styles-inline-css' type='text/css'>
:root{--wp--preset--aspect-ratio--square: 1;--wp--preset--aspect-ratio--4-3: 4/3;--wp--preset--aspect-ratio--3-4: 3/4;--wp--preset--aspect-ratio--3-2: 3/2;--wp--preset--aspect-ratio--2-3: 2/3;--wp--preset--aspect-ratio--16-9: 16/9;--wp--preset--aspect-ratio--9-16: 9/16;--wp--preset--color--black: #000000;--wp--preset--color--cyan-bluish-gray: #abb8c3;--wp--preset--color--white: #ffffff;--wp--preset--color--pale-pink: #f78da7;--wp--preset--color--vivid-red: #cf2e2e;--wp--preset--color--luminous-vivid-orange: #ff6900;--wp--preset--color--luminous-vivid-amber: #fcb900;--wp--preset--color--light-green-cyan: #7bdcb5;--wp--preset--color--vivid-green-cyan: #00d084;--wp--preset--color--pale-cyan-blue: #8ed1fc;--wp--preset--color--vivid-cyan-blue: #0693e3;--wp--preset--color--vivid-purple: #9b51e0;--wp--preset--gradient--vivid-cyan-blue-to-vivid-purple: linear-gradient(135deg,rgb(6,147,227) 0%,rgb(155,81,224) 100%);--wp--preset--gradient--light-green-cyan-to-vivid-green-cyan: linear-gradient(135deg,rgb(122,220,180) 0%,rgb(0,208,130) 100%);--wp--preset--gradient--luminous-vivid-amber-to-luminous-vivid-orange: linear-gradient(135deg,rgb(252,185,0) 0%,rgb(255,105,0) 100%);--wp--preset--gradient--luminous-vivid-orange-to-vivid-red: linear-gradient(135deg,rgb(255,105,0) 0%,rgb(207,46,46) 100%);--wp--preset--gradient--very-light-gray-to-cyan-bluish-gray: linear-gradient(135deg,rgb(238,238,238) 0%,rgb(169,184,195) 100%);--wp--preset--gradient--cool-to-warm-spectrum: linear-gradient(135deg,rgb(74,234,220) 0%,rgb(151,120,209) 20%,rgb(207,42,186) 40%,rgb(238,44,130) 60%,rgb(251,105,98) 80%,rgb(254,248,76) 100%);--wp--preset--gradient--blush-light-purple: linear-gradient(135deg,rgb(255,206,236) 0%,rgb(152,150,240) 100%);--wp--preset--gradient--blush-bordeaux: linear-gradient(135deg,rgb(254,205,165) 0%,rgb(254,45,45) 50%,rgb(107,0,62) 100%);--wp--preset--gradient--luminous-dusk: linear-gradient(135deg,rgb(255,203,112) 0%,rgb(199,81,192) 50%,rgb(65,88,208) 100%);--wp--preset--gradient--pale-ocean: linear-gradient(135deg,rgb(255,245,203) 0%,rgb(182,227,212) 50%,rgb(51,167,181) 100%);--wp--preset--gradient--electric-grass: linear-gradient(135deg,rgb(202,248,128) 0%,rgb(113,206,126) 100%);--wp--preset--gradient--midnight: linear-gradient(135deg,rgb(2,3,129) 0%,rgb(40,116,252) 100%);--wp--preset--font-size--small: 13px;--wp--preset--font-size--medium: 20px;--wp--preset--font-size--large: 36px;--wp--preset--font-size--x-large: 42px;--wp--preset--spacing--20: 0.44rem;--wp--preset--spacing--30: 0.67rem;--wp--preset--spacing--40: 1rem;--wp--preset--spacing--50: 1.5rem;--wp--preset--spacing--60: 2.25rem;--wp--preset--spacing--70: 3.38rem;--wp--preset--spacing--80: 5.06rem;--wp--preset--shadow--natural: 6px 6px 9px rgba(0, 0, 0, 0.2);--wp--preset--shadow--deep: 12px 12px 50px rgba(0, 0, 0, 0.4);--wp--preset--shadow--sharp: 6px 6px 0px rgba(0, 0, 0, 0.2);--wp--preset--shadow--outlined: 6px 6px 0px -3px rgb(255, 255, 255), 6px 6px rgb(0, 0, 0);--wp--preset--shadow--crisp: 6px 6px 0px rgb(0, 0, 0);}:where(.is-layout-flex){gap: 0.5em;}:where(.is-layout-grid){gap: 0.5em;}body .is-layout-flex{display: flex;}.is-layout-flex{flex-wrap: wrap;align-items: center;}.is-layout-flex > :is(*, div){margin: 0;}body .is-layout-grid{display: grid;}.is-layout-grid > :is(*, div){margin: 0;}:where(.wp-block-columns.is-layout-flex){gap: 2em;}:where(.wp-block-columns.is-layout-grid){gap: 2em;}:where(.wp-block-post-template.is-layout-flex){gap: 1.25em;}:where(.wp-block-post-template.is-layout-grid){gap: 1.25em;}.has-black-color{color: var(--wp--preset--color--black) !important;}.has-cyan-bluish-gray-color{color: var(--wp--preset--color--cyan-bluish-gray) !important;}.has-white-color{color: var(--wp--preset--color--white) !important;}.has-pale-pink-color{color: var(--wp--preset--color--pale-pink) !important;}.has-vivid-red-color{color: var(--wp--preset--color--vivid-red) !important;}.has-luminous-vivid-orange-color{color: var(--wp--preset--color--luminous-vivid-orange) !important;}.has-luminous-vivid-amber-color{color: var(--wp--preset--color--luminous-vivid-amber) !important;}.has-light-green-cyan-color{color: var(--wp--preset--color--light-green-cyan) !important;}.has-vivid-green-cyan-color{color: var(--wp--preset--color--vivid-green-cyan) !important;}.has-pale-cyan-blue-color{color: var(--wp--preset--color--pale-cyan-blue) !important;}.has-vivid-cyan-blue-color{color: var(--wp--preset--color--vivid-cyan-blue) !important;}.has-vivid-purple-color{color: var(--wp--preset--color--vivid-purple) !important;}.has-black-background-color{background-color: var(--wp--preset--color--black) !important;}.has-cyan-bluish-gray-background-color{background-color: var(--wp--preset--color--cyan-bluish-gray) !important;}.has-white-background-color{background-color: var(--wp--preset--color--white) !important;}.has-pale-pink-background-color{background-color: var(--wp--preset--color--pale-pink) !important;}.has-vivid-red-background-color{background-color: var(--wp--preset--color--vivid-red) !important;}.has-luminous-vivid-orange-background-color{background-color: var(--wp--preset--color--luminous-vivid-orange) !important;}.has-luminous-vivid-amber-background-color{background-color: var(--wp--preset--color--luminous-vivid-amber) !important;}.has-light-green-cyan-background-color{background-color: var(--wp--preset--color--light-green-cyan) !important;}.has-vivid-green-cyan-background-color{background-color: var(--wp--preset--color--vivid-green-cyan) !important;}.has-pale-cyan-blue-background-color{background-color: var(--wp--preset--color--pale-cyan-blue) !important;}.has-vivid-cyan-blue-background-color{background-color: var(--wp--preset--color--vivid-cyan-blue) !important;}.has-vivid-purple-background-color{background-color: var(--wp--preset--color--vivid-purple) !important;}.has-black-border-color{border-color: var(--wp--preset--color--black) !important;}.has-cyan-bluish-gray-border-color{border-color: var(--wp--preset--color--cyan-bluish-gray) !important;}.has-white-border-color{border-color: var(--wp--preset--color--white) !important;}.has-pale-pink-border-color{border-color: var(--wp--preset--color--pale-pink) !important;}.has-vivid-red-border-color{border-color: var(--wp--preset--color--vivid-red) !important;}.has-luminous-vivid-orange-border-color{border-color: var(--wp--preset--color--luminous-vivid-orange) !important;}.has-luminous-vivid-amber-border-color{border-color: var(--wp--preset--color--luminous-vivid-amber) !important;}.has-light-green-cyan-border-color{border-color: var(--wp--preset--color--light-green-cyan) !important;}.has-vivid-green-cyan-border-color{border-color: var(--wp--preset--color--vivid-green-cyan) !important;}.has-pale-cyan-blue-border-color{border-color: var(--wp--preset--color--pale-cyan-blue) !important;}.has-vivid-cyan-blue-border-color{border-color: var(--wp--preset--color--vivid-cyan-blue) !important;}.has-vivid-purple-border-color{border-color: var(--wp--preset--color--vivid-purple) !important;}.has-vivid-cyan-blue-to-vivid-purple-gradient-background{background: var(--wp--preset--gradient--vivid-cyan-blue-to-vivid-purple) !important;}.has-light-green-cyan-to-vivid-green-cyan-gradient-background{background: var(--wp--preset--gradient--light-green-cyan-to-vivid-green-cyan) !important;}.has-luminous-vivid-amber-to-luminous-vivid-orange-gradient-background{background: var(--wp--preset--gradient--luminous-vivid-amber-to-luminous-vivid-orange) !important;}.has-luminous-vivid-orange-to-vivid-red-gradient-background{background: var(--wp--preset--gradient--luminous-vivid-orange-to-vivid-red) !important;}.has-very-light-gray-to-cyan-bluish-gray-gradient-background{background: var(--wp--preset--gradient--very-light-gray-to-cyan-bluish-gray) !important;}.has-cool-to-warm-spectrum-gradient-background{background: var(--wp--preset--gradient--cool-to-warm-spectrum) !important;}.has-blush-light-purple-gradient-background{background: var(--wp--preset--gradient--blush-light-purple) !important;}.has-blush-bordeaux-gradient-background{background: var(--wp--preset--gradient--blush-bordeaux) !important;}.has-luminous-dusk-gradient-background{background: var(--wp--preset--gradient--luminous-dusk) !important;}.has-pale-ocean-gradient-background{background: var(--wp--preset--gradient--pale-ocean) !important;}.has-electric-grass-gradient-background{background: var(--wp--preset--gradient--electric-grass) !important;}.has-midnight-gradient-background{background: var(--wp--preset--gradient--midnight) !important;}.has-small-font-size{font-size: var(--wp--preset--font-size--small) !important;}.has-medium-font-size{font-size: var(--wp--preset--font-size--medium) !important;}.has-large-font-size{font-size: var(--wp--preset--font-size--large) !important;}.has-x-large-font-size{font-size: var(--wp--preset--font-size--x-large) !important;}
/*# sourceURL=global-styles-inline-css */
</style>

<link rel='stylesheet' id='usc-style-adobe-fonts-css' href='https://use.typekit.net/bfk1gyk.css?ver=a292686bb14d2fb2e3065f' type='text/css' media='all' />
<link rel='stylesheet' id='usc-style-fonts-css' href='https://fonts.googleapis.com/css2?family=Source+Sans+Pro%3Aital%2Cwght%400%2C300%3B0%2C400%3B0%2C600%3B0%2C700%3B0%2C900%3B1%2C400%3B1%2C600%3B1%2C700%3B1%2C900&#038;display=swap&#038;ver=a292686bb14d2fb2e3065f' type='text/css' media='all' />
<link rel='stylesheet' id='usc-style-pattern-lab-css' href='https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/css/style.css?ver=a292686bb14d2fb2e3065f' type='text/css' media='all' />
<link rel='stylesheet' id='usc-style-lity-css' href='https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/css/vendor/lity.min.css?ver=a292686bb14d2fb2e3065f' type='text/css' media='all' />
<link rel='stylesheet' id='usc-style-swiper-css' href='https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/css/vendor/swiper-bundle.min.css?ver=a292686bb14d2fb2e3065f' type='text/css' media='all' />

<!-- Google tag (gtag.js) snippet added by Site Kit -->
<!-- Google Analytics snippet added by Site Kit -->
<script type="text/javascript" src="https://www.googletagmanager.com/gtag/js?id=GT-PB6MQTV" id="google_gtagjs-js" async></script>
<script type="text/javascript" id="google_gtagjs-js-after">
/* <![CDATA[ */
window.dataLayer = window.dataLayer || [];function gtag(){dataLayer.push(arguments);}
gtag("set","linker",{"domains":["today.usc.edu"]});
gtag("js", new Date());
gtag("set", "developer_id.dZTNiMT", true);
gtag("config", "GT-PB6MQTV");
//# sourceURL=google_gtagjs-js-after
/* ]]> */
</script>
<link rel="https://api.w.org/" href="https://today.usc.edu/wp-json/" /><link rel="alternate" title="JSON" type="application/json" href="https://today.usc.edu/wp-json/wp/v2/pages/1271" /><link rel="EditURI" type="application/rsd+xml" title="RSD" href="https://today.usc.edu/xmlrpc.php?rsd" />
<meta name="generator" content="WordPress 6.9.4" />
<link rel='shortlink' href='https://today.usc.edu/?p=1271' />
<meta name="generator" content="Site Kit by Google 1.176.0" />
<!-- Google Tag Manager snippet added by Site Kit -->
<script type="text/javascript">
/* <![CDATA[ */

			( function( w, d, s, l, i ) {
				w[l] = w[l] || [];
				w[l].push( {'gtm.start': new Date().getTime(), event: 'gtm.js'} );
				var f = d.getElementsByTagName( s )[0],
					j = d.createElement( s ), dl = l != 'dataLayer' ? '&l=' + l : '';
				j.async = true;
				j.src = 'https://www.googletagmanager.com/gtm.js?id=' + i + dl;
				f.parentNode.insertBefore( j, f );
			} )( window, document, 'script', 'dataLayer', 'GTM-NZS3SS3D' );
			
/* ]]> */
</script>

<!-- End Google Tag Manager snippet added by Site Kit -->
<link rel="icon" href="https://today.usc.edu/wp-content/uploads/2025/11/cropped-usc_512x512-Android@1x-32x32.png" sizes="32x32" />
<link rel="icon" href="https://today.usc.edu/wp-content/uploads/2025/11/cropped-usc_512x512-Android@1x-192x192.png" sizes="192x192" />
<link rel="apple-touch-icon" href="https://today.usc.edu/wp-content/uploads/2025/11/cropped-usc_512x512-Android@1x-180x180.png" />
<meta name="msapplication-TileImage" content="https://today.usc.edu/wp-content/uploads/2025/11/cropped-usc_512x512-Android@1x-270x270.png" />

			</head>

	<body class="" page-id="1271" data-template="base.twig">

		
							






		


				

	<div role="region" aria-label="Skip to main">
		<a class="skip-link" href="#main-content">Skip to Content</a>
	</div>

	<header id="l--main-header" class="header" aria-label="Site Header">

					<div class="r--region r--header">
									  
    






<div
  class="cc--component-container cc--header gold"

  
  
  
  
  
  
  >
  <div class="c--component c--header"
    
      >

            <div class="alerts-wrapper">
      <div class="alert-banner-block-container"></div>
      <div class="page-alert-banner-block-container"></div>
    </div>
  
  
  <div class="header-wrapper">
    <div class="desktop-nav">
        
    






<div
  class="cc--component-container cc--branding-header gold"

  
  
  
  
  
  
  >
  <div class="c--component c--branding-header"
    
      >

    
  <div class="logo-wrapper">
    <a href="https://www.usc.edu" aria-label="University of Southern California" rel="home">
      <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
        viewBox="0 0 178 58" style="enable-background:new 0 0 178 58;" xml:space="preserve">
        <path class="red white" d="M59.8,11.2c0-2.3-0.5-3.3-2.3-3.3h-3.1c-0.4,0-0.5-0.1-0.5-0.6V5.8c0-0.4,0.1-0.5,0.7-0.5
          c0.7,0,3.9,0.2,10.3,0.2c5.6,0,8.8-0.2,9.6-0.2c0.4,0,0.5,0.1,0.5,0.5v1.7c0,0.4-0.1,0.5-0.5,0.5h-2.8c-2,0-3.2,1.2-3.3,2.6
          c-0.1,0.8-0.2,4.8-0.2,8.7v8.3c0,3.3,0.1,8.1,0.2,10.2c0.2,3.1,0.7,6.1,2.3,8.3c2.3,3.1,5.9,4.1,9.1,4.1c3.1,0,6.8-1.2,9-3.2
          c2.5-2.5,3.9-6.4,3.9-11.3v-12c0-5.3-0.5-11.2-0.7-12.8c-0.2-1.7-1.3-2.9-2.7-2.9h-2.4c-0.4,0-0.5-0.2-0.5-0.5V5.8
          c0-0.4,0.1-0.5,0.5-0.5c0.7,0,3.7,0.2,8.1,0.2c4.1,0,6.3-0.2,6.8-0.2c0.6,0,0.7,0.2,0.7,0.5v1.5c0,0.4-0.1,0.6-0.4,0.6h-2.4
          c-2.4,0-2.7,1.9-2.8,2.6c-0.1,0.8-0.1,10.2-0.1,11c0,0.9-0.1,12.6-0.4,15.2c-0.4,5.5-2,9.6-5.6,12.8c-2.8,2.5-7.1,4.1-12.2,4.1
          c-3.2,0-7.1-0.5-9.6-1.6c-3.9-1.6-6.5-4.7-7.9-9c-0.9-2.9-1.2-7.3-1.2-16.1V11.2z"/>
        <path class="red white" d="M104.3,37.8c0.3,0,0.4,0.1,0.4,0.5c0.1,2.1,0.7,4.7,2.5,7.6c2,3.2,5.5,4.8,8.8,4.8
          c3.6,0,8.7-2.3,8.7-7.8c0-6.1-2.8-8.1-9.9-10.8c-2.1-0.8-6.5-2.7-9.8-5.6c-2.3-2.1-3.3-5.5-3.3-9.6c0-3.7,1.5-6.8,4.7-9.4
          c2.7-2.3,5.6-3.2,9.5-3.2c4.4,0,7.1,1.1,8.4,1.6c0.7,0.3,1.1,0.1,1.4-0.2l0.9-0.9c0.3-0.3,0.5-0.4,0.9-0.4c0.4,0,0.5,0.4,0.5,1.1
          c0,1.1,1.1,10.4,1.1,11.1c0,0.3-0.1,0.4-0.4,0.5l-1.1,0.3c-0.3,0.1-0.5,0-0.7-0.4c-0.7-1.9-1.7-4.5-4.1-7.1
          c-1.7-1.9-3.9-2.8-6.9-2.8c-5.3,0-7.5,4.3-7.5,6.7c0,2.3,0.3,4.3,1.9,6c1.7,1.7,5.5,3.3,8.1,4.1c3.9,1.2,7.1,2.5,9.6,5.2
          c2.1,2.3,3.3,5.1,3.3,9.6c0,8.1-5.3,14.8-14.2,14.8c-5.5,0-8.7-1.2-11.4-3.2c-0.8-0.5-1.2-0.8-1.6-0.8c-0.4,0-0.7,0.5-1.1,1.2
          c-0.4,0.7-0.7,0.8-1.2,0.8c-0.5,0-0.8-0.4-0.8-1.1c0.1-0.8,1.1-11.6,1.2-12.4c0.1-0.4,0.1-0.5,0.4-0.5L104.3,37.8z"/>
        <path class="red white" d="M177.1,41.8c0.2,0.2,0.3,0.4,0.1,0.7c-4,5.7-10.4,11.2-19.9,11.2c-8.6,0-13.4-2.5-17.5-6.3
          c-4.9-4.4-7.2-12.2-7.2-18c0-6.5,2.1-12.8,6.8-17.9c3.6-3.9,9.3-7.1,17.4-7.1c6.1,0,11.1,1.9,13.5,3.2c0.9,0.5,1.2,0.5,1.3,0
          l0.5-1.7c0.1-0.4,0.3-0.4,1.1-0.4c0.8,0,0.9,0.1,0.9,1.1c0,1.2,1.3,11.9,1.6,13.9c0.1,0.4,0,0.6-0.3,0.7l-1.3,0.4
          c-0.3,0.1-0.5-0.1-0.7-0.7c-0.8-2.4-2.4-5.7-5.2-8.7c-2.9-2.8-6.4-4.9-11.6-4.9c-10.7,0-15.1,9.6-15.1,19.2c0,4.1,1.2,12,6.1,17.4
          c4.4,4.7,8.3,5.2,11.5,5.2c7.3,0,13.2-3.7,16-7.7c0.3-0.5,0.6-0.7,0.8-0.5L177.1,41.8z"/>
        <path class="black white" d="M39.7,35.3C37,41.9,31.8,51,22.4,56.7C13,51,7.8,42,5.1,35.4c-2.5-6.1-3.4-11.3-3.7-12.8
          c1.1-0.6,2.2-0.9,3.5-0.9h0c0.9,0,2.1,0.2,2.9,0.4l0.4,0.1c0.4,0.1,0.8,0.2,1.2,0.3c1.4,0.4,3,0.8,4.1,0.8h0c1.3,0,3.4-0.2,5.7-1
          c0.7-0.2,1.5-0.5,2.3-0.6l0.8-0.1h0l1.2,0.1c0.7,0.1,1.4,0.3,2,0.5c2.3,0.8,4.4,1,5.7,1h0c0.8,0,1.8-0.2,3.3-0.6
          c0.6-0.2,1.2-0.3,1.8-0.5l0.7-0.2c0.7-0.2,2-0.4,2.9-0.4h0c1.3,0,2.4,0.3,3.5,0.9C43.1,24,42.2,29.2,39.7,35.3L39.7,35.3z M0.8,12.7
          l13.7,3c-0.1,0.2-0.2,0.4-0.3,0.6L0.8,14.6C0.8,14,0.8,13.3,0.8,12.7L0.8,12.7z M1.3,7.7l14.2,6.6c-0.3,0.3-0.5,0.6-0.7,1L0.9,11.4
          C1,10.2,1.1,9,1.3,7.7L1.3,7.7z M1.7,5.6c0.5,0.1,1.6,0.3,3,0.5l11.5,7.4c-0.2,0.1-0.3,0.3-0.4,0.4L1.5,6.3C1.6,6.1,1.6,5.8,1.7,5.6
          L1.7,5.6z M7.8,6.2c1.5,0,2.9-0.1,4.2-0.4l5.8,6.5c-0.4,0.3-0.9,0.5-1.2,0.9l-9.6-7C7.2,6.2,7.5,6.2,7.8,6.2L7.8,6.2z M15.2,5
          l3.9,6.8c-0.3,0.1-0.5,0.2-0.8,0.3l-4.9-6.6C14,5.4,14.6,5.2,15.2,5L15.2,5z M20.3,2.5l1.2,8.7c-0.7,0.1-1.3,0.2-1.9,0.4l-3.2-7
          C18,3.9,19.3,3.2,20.3,2.5L20.3,2.5z M22.3,1c0.3,0.2,0.6,0.5,0.9,0.7l-0.5,9.4c-0.1,0-0.2,0-0.3,0h0c-0.1,0-0.2,0-0.3,0l-0.5-9.5
          C21.9,1.4,22.2,1.2,22.3,1L22.3,1z M24.4,2.5c1.4,0.9,2.7,1.6,3.9,2l-3.2,7c-0.6-0.2-1.2-0.3-1.9-0.4L24.4,2.5z M29.4,5
          c0.8,0.3,1.5,0.5,1.8,0.5l-4.9,6.6c-0.2-0.1-0.5-0.2-0.8-0.3L29.4,5z M32.7,5.8c1.4,0.3,2.9,0.4,4.5,0.4c0.2,0,0.4,0,0.5,0l-9.6,7
          c-0.4-0.3-0.8-0.6-1.2-0.9L32.7,5.8z M39.9,6.1c1.5-0.1,2.5-0.4,3.1-0.5c0.1,0.2,0.1,0.5,0.1,0.7l-14.3,7.6
          c-0.1-0.2-0.3-0.3-0.5-0.4L39.9,6.1z M43.4,7.7c0.2,1.3,0.3,2.6,0.4,3.8L30,15.3c-0.2-0.4-0.5-0.7-0.8-1.1L43.4,7.7z M43.9,14.6
          l-13.3,1.8c-0.1-0.2-0.2-0.4-0.3-0.6l13.6-3C43.9,13.4,43.9,14,43.9,14.6L43.9,14.6z M39.9,17.4c-0.8,0-2.2,0.2-3.1,0.4L36.1,18
          c-0.6,0.2-1.2,0.4-1.8,0.5c-0.8,0.2-2,0.5-2.9,0.5c-0.1-0.8-0.3-1.5-0.6-2.3l13.1-1c0,0.9-0.1,1.8-0.1,2.5
          C42.6,17.8,41.2,17.5,39.9,17.4L39.9,17.4z M39.8,19.1L39.8,19.1c-0.9,0-2.3,0.2-3.1,0.4l-0.6,0.2c-0.6,0.2-1.2,0.4-1.8,0.5l0.1,0.4
          l-0.1-0.4c-0.9,0.2-2.1,0.5-3.1,0.5h0c-1.3,0-3.2-0.2-5.5-0.9c-0.6-0.2-1.3-0.4-2.1-0.6c-0.4-0.1-0.9-0.1-1.3-0.1
          c-0.3,0-0.6,0-0.9,0.1c-0.9,0.1-1.8,0.4-2.5,0.6c-2.2,0.8-4.2,0.9-5.5,1h0c-1.1,0-2.6-0.4-3.9-0.8L8,19.6c-0.8-0.2-2.1-0.4-3.1-0.4
          h0c-1.3,0-2.6,0.3-3.8,0.9C1,19.8,1,19.6,1,19.3c1.2-0.6,2.6-1,4-1h0c0.9,0,2.1,0.2,2.9,0.4l1.6,0.5l0.1-0.4l-0.1,0.4
          c1.5,0.4,3,0.8,4.1,0.8h0c1.3,0,3.4-0.2,5.7-1c0.7-0.2,1.5-0.5,2.3-0.6l0.8-0.1l1.2,0.1c0.7,0.1,1.4,0.4,2,0.5c2.3,0.8,4.4,1,5.7,1
          h0c0.8,0,1.8-0.2,3.3-0.6c0.6-0.2,1.2-0.3,1.8-0.5l0.6-0.2c0.7-0.2,2-0.4,2.9-0.4h0c1.3,0,2.7,0.3,3.9,0.9c0,0.3-0.1,0.6-0.1,0.9
          C42.5,19.4,41.1,19.1,39.8,19.1L39.8,19.1z M22.4,17.5L22.4,17.5L22.4,17.5c-0.3,0-0.6,0-0.9,0.1c-0.9,0.1-1.8,0.4-2.5,0.6
          c-1.9,0.7-3.6,0.9-4.9,0.9c0.6-4.1,4-7.2,8.2-7.2h0c4.1,0,7.7,3.1,8.2,7.1c-1.2-0.1-2.9-0.3-4.8-0.9c-0.6-0.2-1.3-0.4-2.1-0.6l0,0
          l0,0C23.2,17.5,22.8,17.5,22.4,17.5L22.4,17.5z M0.8,15.8l13.1,1c-0.3,0.7-0.5,1.5-0.6,2.3c-1.1,0-2.5-0.4-3.7-0.8L8,17.9
          c-0.8-0.2-2.1-0.4-3.1-0.4h0c-1.4,0-2.8,0.3-4.1,0.9C0.9,17.7,0.8,16.8,0.8,15.8L0.8,15.8z M39.8,20.8L39.8,20.8
          c-0.9,0-2.3,0.2-3.1,0.4L36,21.4c-0.6,0.2-1.2,0.3-1.7,0.5l0.1,0.4l-0.1-0.4c-1.4,0.4-2.3,0.5-3.1,0.5h0c-1.3,0-3.2-0.2-5.5-0.9
          c-0.6-0.2-1.3-0.4-2.1-0.6c-0.8-0.1-1.5-0.2-2.2-0.1c-0.9,0.1-1.8,0.4-2.5,0.6c-2.2,0.8-4.2,0.9-5.5,1h0c-1.1,0-2.6-0.4-3.9-0.8
          l-0.1,0.4l0.1-0.4c-0.4-0.1-0.8-0.2-1.2-0.3L8,21.3c-0.8-0.2-2.1-0.4-3.1-0.4h0c-1.3,0-2.5,0.3-3.6,0.9c0-0.2-0.1-0.5-0.1-0.8
          c1.1-0.6,2.4-1,3.7-1h0c0.9,0,2.1,0.2,2.9,0.4l1.6,0.5l0.1-0.4l-0.1,0.4c1.5,0.4,3,0.8,4.1,0.8h0c1.3,0,3.4-0.2,5.7-1
          c0.7-0.2,1.5-0.5,2.3-0.6l0.8-0.1h0l1.2,0.1c0.7,0.1,1.4,0.4,2,0.5c2.3,0.8,4.4,1,5.7,1h0c0.8,0,1.8-0.2,3.3-0.6
          c0.6-0.2,1.2-0.3,1.8-0.5l0.6-0.2c0.7-0.2,2-0.4,2.9-0.4h0c1.3,0,2.6,0.3,3.8,1c0,0.3-0.1,0.6-0.1,0.8
          C42.3,21.1,41.1,20.8,39.8,20.8L39.8,20.8z M44.7,12.6l-0.1-1.3c-0.1-1.2-0.2-2.6-0.5-3.9L44,6.4c-0.1-0.5-0.2-1-0.3-1.5l-0.1-0.4
          l-0.4,0.1c0,0-0.7,0.2-1.7,0.4l-2.6,0.3c-0.5,0-1.1,0.1-1.7,0.1c-1.5,0-3-0.1-4.5-0.4l-0.9-0.2c0,0,0,0,0,0c0,0-0.8-0.2-1.9-0.5
          l-1.2-0.5c-1.2-0.5-2.7-1.2-4.1-2.2l-1.2-0.9c-0.2-0.2-0.4-0.4-0.7-0.6L22.3,0l-0.3,0.2c0,0-0.2,0.2-0.6,0.5l-1.2,0.9
          c-1,0.7-2.4,1.5-4.2,2.2l-1.2,0.5c-0.6,0.2-1.2,0.4-1.9,0.6c0,0,0,0,0,0l-1.1,0.2c-1.3,0.2-2.6,0.4-4,0.4c-0.7,0-1.4,0-2-0.1
          L3.2,5.1c-1-0.2-1.7-0.3-1.7-0.3L1.1,4.6L1,5C0.9,5.4,0.8,5.8,0.8,6.2L0.6,7.4C0.4,8.7,0.2,10,0.1,11.2L0,12.5c0,0.7,0,1.3,0,1.9
          l0,1.3c0.1,3.7,0.5,6.2,0.5,6.4c0,0.3,0.8,6.3,3.8,13.4c2.8,6.7,8.1,15.9,17.7,21.7l0.4,0.3l0.4-0.3c9.6-5.8,14.8-15.1,17.6-21.8
          c2.9-7.2,3.7-13.1,3.7-13.4c0-0.2,0.5-2.7,0.5-6.4l0-1.3C44.7,13.9,44.7,13.2,44.7,12.6L44.7,12.6z"/>
        <path class="black white" d="M33.5,31c-0.1,0.1-0.3,0.2-0.4,0.2l-3,0c-0.4,0-0.6-0.3-0.6-0.6c0-0.3,0.3-0.6,0.6-0.6l0.5,0
          c0,0,0,0,0,0l0,0h0.6l1.8,0c0.3,0,0.6,0.3,0.6,0.6C33.7,30.7,33.6,30.9,33.5,31L33.5,31z M32.9,32.1c-0.1,0.1-0.3,0.2-0.5,0.2
          l-1.8,0c-0.3,0-0.5-0.2-0.6-0.6c0,0,0,0,0,0l3,0c0,0,0,0,0,0C33,31.8,33,32,32.9,32.1L32.9,32.1z M32.1,39.7l-1.2,0l-0.3-7l1.8,0
          c0,0,0,0,0,0L32.1,39.7z M32.9,40.7c0,0.3-0.2,0.5-0.5,0.5l-1.7,0l0,0.2v-0.2c-0.3,0-0.5-0.2-0.5-0.5c0-0.3,0.2-0.5,0.5-0.5l1,0
          l0.8,0C32.7,40.2,32.9,40.4,32.9,40.7L32.9,40.7z M31.5,45.1c-0.2,0-0.4-0.2-0.4-0.4l-0.1-3h1l-0.1,3C31.9,44.9,31.7,45.1,31.5,45.1
          L31.5,45.1z M29.1,28.4c0-0.4,0.2-0.7,0.3-1.1c0.2-0.4,0.4-0.9,0.3-1.4c0.3,0.1,0.7,0.3,0.9,0.6c0,0,0,0,0,0
          c-0.1,0.1-0.1,0.2-0.2,0.2c0,0-1.2,1.2-0.4,2.7H30c-0.1,0-0.2,0-0.3,0C29.7,29.4,29.1,29.2,29.1,28.4L29.1,28.4z M31.7,25.9
          c0.1-0.2,0.1-0.4,0-0.6c0.5,0.4,1.1,1.1,1.1,2c0,1-0.7,1.9-0.9,2.1h-0.2c-0.2-0.3-0.6-0.9-0.2-1.7c0.1-0.1,0.5-0.8,0.5-0.9
          c0.1-0.1,0.1-0.3,0.1-0.3c-0.1,0.1-0.2,0.2-0.2,0.2c-0.1,0.1-0.5,0.5-0.6,0.7c-0.6,0.8-0.2,1.6,0,1.9l-0.6,0
          c-0.9-1.3,0.1-2.3,0.1-2.3C31,27,31,27,31.1,26.9C31.4,26.6,31.6,26.4,31.7,25.9L31.7,25.9z M33.1,26.3c0.2-0.2,0.4-0.4,0.5-0.5
          c-0.1,1.1,0.4,2.2,0.5,2.2c0.3,0.9-0.7,1.4-0.9,1.5c-0.1,0-0.2,0-0.3,0h-0.5c0.3-0.5,0.8-1.2,0.8-2.1C33.3,26.9,33.1,26.3,33.1,26.3
          L33.1,26.3z M33.8,29.7c0.5-0.3,1.1-1,0.8-1.9c0,0-0.7-1.3-0.3-2.3l0.1-0.3l-0.4,0c-0.1,0-0.5,0.1-1.1,0.7c-0.6-0.9-1.5-1.5-1.5-1.5
          l-0.9-0.5l0.6,0.9c0.1,0.2,0.4,0.7,0.3,1c-0.1,0.2-0.1,0.3-0.2,0.5c-0.6-0.6-1.5-0.9-1.6-0.9L29,25.2l0.2,0.4c0.3,0.6,0.1,1-0.2,1.5
          c-0.2,0.4-0.4,0.8-0.4,1.3c0,0.7,0.3,1.1,0.6,1.3c-0.2,0.2-0.4,0.5-0.4,0.8c0,0.4,0.2,0.8,0.6,1c0.1,0.6,0.3,0.9,0.6,1.1l0.3,7.2
          c-0.4,0.1-0.8,0.5-0.8,0.9c0,0.5,0.4,0.9,0.8,1l0.1,3c0,0.5,0.4,0.8,0.8,0.8c0.5,0,0.8-0.4,0.8-0.8l0.1-3c0.5-0.1,0.9-0.5,0.9-1
          c0-0.5-0.3-0.9-0.8-0.9l0.3-7.2c0.1,0,0.2-0.1,0.3-0.2c0.2-0.2,0.3-0.5,0.4-0.9c0.1,0,0.2-0.1,0.2-0.2c0.2-0.2,0.3-0.5,0.3-0.8
          C34.1,30.2,34,29.9,33.8,29.7L33.8,29.7z"/>
        <path class="black white" d="M14.5,31.2l-3,0c-0.4,0-0.6-0.3-0.6-0.6c0-0.2,0.1-0.3,0.2-0.4c0.1-0.1,0.3-0.2,0.4-0.2H12
          c0,0,0,0,0,0l0,0h0.6l0.9,0h0.9c0.3,0,0.6,0.3,0.6,0.6C15.1,30.9,14.8,31.2,14.5,31.2L14.5,31.2z M14.3,32.1
          c-0.1,0.1-0.3,0.2-0.5,0.2l-1.8,0c-0.3,0-0.5-0.2-0.6-0.6c0,0,0,0,0,0l3,0c0,0,0,0,0.1,0C14.5,31.9,14.4,32,14.3,32.1L14.3,32.1z
          M13.6,39.8l-1.2,0l-0.3-7h0l1.8,0c0,0,0,0,0,0L13.6,39.8z M14.4,40.8c0,0.1-0.1,0.3-0.2,0.4c-0.1,0.1-0.2,0.2-0.4,0.2h-1.7l0,0.2
          v-0.2c-0.3,0-0.5-0.2-0.5-0.5c0-0.3,0.2-0.5,0.5-0.5h1l0.7,0C14.1,40.2,14.4,40.5,14.4,40.8L14.4,40.8z M13,45.1
          c-0.2,0-0.4-0.2-0.4-0.4l-0.1-3l1,0l-0.1,3C13.4,44.9,13.2,45.1,13,45.1L13,45.1z M10.6,28.5c0-0.4,0.2-0.7,0.3-1.1
          c0.2-0.4,0.4-0.9,0.3-1.4c0.3,0.1,0.7,0.3,0.9,0.6c0,0,0,0,0,0c-0.1,0.1-0.1,0.1-0.2,0.2c0,0-1.2,1.2-0.4,2.7h-0.1
          c-0.1,0-0.1,0-0.2,0C11.2,29.5,10.6,29.2,10.6,28.5L10.6,28.5z M13.2,25.9c0.1-0.2,0.1-0.4,0-0.6c0.5,0.4,1.1,1.1,1.1,2
          c0,1-0.7,1.9-0.9,2.1h-0.2c-0.2-0.3-0.6-0.9-0.2-1.7c0.1-0.1,0.5-0.8,0.5-0.9c0.1-0.1,0.1-0.3,0.1-0.3c-0.1,0.1-0.2,0.2-0.2,0.2
          c-0.1,0.1-0.5,0.5-0.6,0.7c-0.6,0.8-0.2,1.6,0,1.9l-0.6,0c-0.9-1.3,0.1-2.3,0.1-2.3c0.1-0.1,0.1-0.2,0.2-0.2
          C12.8,26.6,13,26.4,13.2,25.9L13.2,25.9z M14.6,26.3c0.2-0.2,0.4-0.4,0.5-0.5C15,26.9,15.6,28,15.6,28c0.3,0.9-0.7,1.4-0.9,1.5
          c-0.1,0-0.2,0-0.3,0H14c0.3-0.5,0.8-1.2,0.8-2.1C14.8,26.9,14.6,26.3,14.6,26.3L14.6,26.3z M15.2,29.8c0.5-0.3,1.1-1,0.8-1.9
          c0,0-0.7-1.3-0.4-2.3l0.1-0.3l-0.4,0c-0.1,0-0.5,0.1-1.1,0.7c-0.6-0.9-1.5-1.5-1.5-1.5l-0.9-0.5l0.6,0.9c0.1,0.2,0.4,0.7,0.3,1
          c-0.1,0.2-0.1,0.3-0.2,0.4c-0.6-0.6-1.5-0.9-1.6-0.9l-0.5-0.1l0.2,0.4c0.3,0.6,0.1,1-0.2,1.5c-0.2,0.4-0.4,0.8-0.4,1.3
          c0,0.7,0.4,1.1,0.6,1.3c0,0,0,0,0,0c-0.2,0.2-0.3,0.5-0.3,0.8c0,0.4,0.2,0.8,0.6,1c0,0.5,0.3,0.9,0.6,1.1l0.3,7.2
          c-0.4,0.1-0.8,0.5-0.8,0.9c0,0.5,0.4,0.9,0.8,1l0.1,3c0,0.5,0.4,0.8,0.8,0.8c0.5,0,0.8-0.4,0.8-0.8l0.1-3c0.2,0,0.4-0.1,0.6-0.3
          c0.2-0.2,0.3-0.4,0.3-0.7c0-0.5-0.3-0.9-0.8-0.9l0.3-7.2c0.1,0,0.2-0.1,0.3-0.2c0.2-0.2,0.3-0.5,0.3-0.9c0.3-0.2,0.6-0.5,0.6-1
          C15.6,30.3,15.4,30,15.2,29.8L15.2,29.8z"/>
        <path class="black white" d="M25.1,32.4c-0.2,0.2-0.4,0.3-0.7,0.3l-4.2,0c-0.5,0-1-0.4-1-1c0-0.5,0.4-1,1-1c0,0,2.8,0,2.8,0
          h1.4c0.5,0,1,0.4,1,1C25.4,32,25.3,32.2,25.1,32.4L25.1,32.4z M23.6,34l-2.5,0h0c-0.4,0-0.8-0.4-0.9-0.9c0,0,0.1,0,0.1,0l4.2,0
          c0.1,0,0.1,0,0.2,0C24.4,33.6,24,34,23.6,34L23.6,34z M23.3,44.6l-1.9,0L21,34.5c0,0,0.1,0,0.1,0h0l2.5,0c0.1,0,0.1,0,0.2,0
          L23.3,44.6z M24.4,45.9c0,0.4-0.4,0.8-0.8,0.8l-2.4,0c-0.4,0-0.8-0.4-0.8-0.8c0-0.4,0.4-0.8,0.8-0.8l1.8,0h0.6
          C24,45.1,24.4,45.5,24.4,45.9L24.4,45.9z M23,51.5c0,0.2-0.1,0.3-0.2,0.4c-0.1,0.1-0.3,0.2-0.4,0.2h0c-0.2,0-0.3-0.1-0.4-0.2
          c-0.1-0.1-0.2-0.3-0.2-0.4l-0.2-4.2l1.6,0L23,51.5z M18.9,28.7c0-0.6,0.2-1,0.5-1.5c0.3-0.6,0.6-1.3,0.4-2.1c0.4,0.2,1,0.4,1.5,0.9
          c-0.1,0.1-0.1,0.1-0.2,0.2c-0.1,0.1-0.2,0.2-0.3,0.3c0,0-1.7,1.7-0.4,3.7h-0.2c-0.1,0-0.2,0-0.4,0.1C19.8,30.3,18.9,29.9,18.9,28.7
          L18.9,28.7z M22.4,25.2c0.1-0.4,0.1-0.8,0-1.1c0.7,0.6,1.8,1.7,1.8,3.1c0,1.5-1,2.7-1.4,3.1h-0.1c-0.2-0.3-0.4-0.8-0.4-1.3
          c0-1,0.5-1.6,0.5-1.6s-0.6,0.4-0.8,1.4c-0.1,0.5,0,0.9,0.3,1.6h-0.6c-0.2-0.3-0.9-1.3-0.2-2.4c0.1-0.2,0.8-1,0.8-1.2
          c0.1-0.2,0.1-0.4,0.1-0.4c-0.1,0.2-0.3,0.3-0.3,0.3c-0.2,0.2-0.7,0.7-0.9,0.9c-1,1.2-0.2,2.5,0,2.8h-0.4c-1.4-1.9,0.1-3.3,0.2-3.4
          c0.1-0.1,0.2-0.2,0.3-0.3C21.8,26.1,22.2,25.8,22.4,25.2L22.4,25.2z M25.4,25.4c-0.3,1.5,0.2,2.7,0.2,2.7c0.4,1.3-0.6,2-0.9,2.2
          c-0.1,0-0.2,0-0.3,0l-0.9,0c0.5-0.6,1.2-1.7,1.2-3.1c0-0.4-0.1-0.7-0.2-1C24.8,25.7,25.2,25.5,25.4,25.4L25.4,25.4z M25.2,30.5
          c0.5-0.4,1.2-1.2,0.8-2.6c0,0-0.5-1.4-0.1-2.9l0.1-0.3l-0.4,0c0,0-0.7,0.1-1.4,0.9c-0.7-1.5-2.3-2.4-2.3-2.5l-0.8-0.5l0.5,0.8
          c0,0,0.5,0.9,0.3,1.5c-0.1,0.3-0.2,0.5-0.3,0.7c-0.8-0.9-2.1-1.2-2.1-1.2L19,24.4l0.2,0.4c0.4,0.9,0.1,1.5-0.3,2.2
          c-0.3,0.5-0.5,1.1-0.5,1.7c0,0.9,0.5,1.5,0.9,1.8c-0.3,0.3-0.6,0.7-0.6,1.1c0,0.6,0.3,1.1,0.8,1.3c0.1,0.5,0.4,1.1,0.9,1.4l0,0
          L21,44.7c-0.6,0.1-1.1,0.6-1.1,1.3c0,0.7,0.5,1.2,1.2,1.3l0.2,4.3c0,0.3,0.1,0.6,0.3,0.8c0.2,0.2,0.5,0.3,0.8,0.3h0
          c0.3,0,0.6-0.1,0.8-0.3c0.2-0.2,0.3-0.5,0.3-0.8l0.2-4.3c0.7,0,1.2-0.6,1.2-1.3c0-0.7-0.5-1.2-1.1-1.3l0.5-10.3h0
          c0.5-0.3,0.8-0.8,0.9-1.3c0.1-0.1,0.2-0.1,0.4-0.3c0.3-0.3,0.4-0.6,0.4-1C25.9,31.2,25.6,30.7,25.2,30.5L25.2,30.5z"/>
      </svg>
    </a>
  </div>

      <div class="site-title">
      <span class="separator"></span>
              <a href="https://today.usc.edu" class="division-link" aria-label="USC Today">USC Today</a>
          </div>
  

  </div></div>

      <div class="header-buttons" role="navigation" aria-label="Search and Navigation">
        <button type="button" class="mobile-menu-trigger" aria-label="Open Site Navigation / Menu" aria-expanded="false" aria-controls="main-navigation">
  <svg version="1.1" class="icon-hamburger" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
	 viewBox="0 0 33 18" style="enable-background:new 0 0 33 18;" xml:space="preserve">
    <path class="black" d="M0,0h32.1v2.6H0V0z M0,7.7h25.7v2.6H0V7.7z M0,15.4h32.1V18H0V15.4z"/>
  </svg>
  <svg height="27" viewBox="0 0 27 27" width="27" aria-label="Close Icon" class="icon-close" xmlns="http://www.w3.org/2000/svg"><path d="m1174.87437 25.1256313c.58579.5857865.58579 1.5355339 0 2.1213204l-10.25308 10.2523398 10.25308 10.2537568c.58579.5857865.58579 1.5355339 0 2.1213204-.58579.5857864-1.53554.5857864-2.12132 0l-10.25376-10.2530772-10.25234 10.2530772c-.58578.5857864-1.53553.5857864-2.12132 0-.58579-.5857865-.58579-1.5355339 0-2.1213204l10.25266-10.2537568-10.25266-10.2523398c-.58579-.5857865-.58579-1.5355339 0-2.1213204.58579-.5857864 1.53554-.5857864 2.12132 0l10.25234 10.2526602 10.25376-10.2526602c.58578-.5857864 1.53553-.5857864 2.12132 0z" fill="#00543c" fill-rule="evenodd" transform="translate(-1149 -24)"/></svg>
  <span><span class="visually-hidden">Open Site Navigation / </span>Menu</span>
</button>

        <button class="icon-search search-trigger-button" aria-label="Open Site Search" aria-expanded="false" aria-controls="search-container">
  <svg class="search-open" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
	 viewBox="0 0 22 22" style="enable-background:new 0 0 22 22;" xml:space="preserve">
    <path class="black" d="M8.8,0c4.9,0,8.8,3.9,8.8,8.8c0,2-0.7,3.9-1.8,5.3l6.2,6.2L20.3,22l-6.2-6.2
    c-1.5,1.1-3.3,1.8-5.3,1.8C3.9,17.6,0,13.7,0,8.8S3.9,0,8.8,0z M8.8,2.4c-3.6,0-6.5,2.9-6.5,6.5s2.9,6.5,6.5,6.5s6.5-2.9,6.5-6.5
    S12.4,2.4,8.8,2.4z"/>
  </svg>
  <svg class="search-close" width="19px" height="19px" viewBox="0 0 19 19" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><g stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"><g transform="translate(-280.000000, -22.000000)" fill="#333333"><g><g transform="translate(274.000000, 15.000000)"><path d="M21.8331075,7.79044769 L23.0452905,9.00263074 L16.6294477,15.4174477 L23.0452905,21.8331075 L21.8331075,23.0452905 L15.4174477,16.6294477 L9.00263074,23.0452905 L7.79044769,21.8331075 L14.2054477,15.4174477 L7.79044769,9.00263074 L9.00263074,7.79044769 L15.4174477,14.2054477 L21.8331075,7.79044769 Z"></path></g></g></g></g></svg>
  <span>Search</span>
</button>
      </div>

      <div class="nav-container" id="main-navigation">
        <div class="nav-wrapper">
            
    






<div
  class="cc--component-container cc--main-menu gold"

  
  
  
  
  
  
  >
  <div class="c--component c--main-menu"
    
      >

    
      <div class="menu-container">
      <nav class="mc--menu">
        <ul class="m--menu">
                                    <li  class="menu-item">
               
  <div class="link-arrow-wrapper">
    <button class="expand" type="button" aria-expanded="false" aria-label="Display Sub Menu for USC News" aria-controls="usc-news32">
      USC News
      <span class="icon">
        <svg width="7px" height="11px" viewBox="0 0 7 11" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><g stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"><g transform="translate(-230.000000, -41.000000)" fill="#333333"><g><g><g transform="translate(0.000000, 24.000000)"><g transform="translate(220.000000, 10.000000)"><polygon transform="translate(13.137755, 12.500000) scale(-1, 1) rotate(90.000000) translate(-13.137755, -12.500000) " points="13.1377551 13.8010204 17.5765306 9.3622449 18.494898 10.2806122 13.1377551 15.6377551 7.78061224 10.2806122 8.69897959 9.3622449"></polygon></g></g></g></g></g></g></svg>
      </span>
    </button>
  </div>


  
  <div class="submenus-wrapper" id="usc-news32">
    <button class="back-to-prev"><svg width="13px" height="12px" viewBox="0 0 13 12" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><g stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"><g transform="translate(-64.000000, -167.000000)" fill="#FFFFFF"><g transform="translate(64.000000, 160.000000)"><g transform="translate(0.000000, 7.000000)"><polygon transform="translate(6.431957, 5.714286) scale(-1, 1) rotate(-270.000000) translate(-6.431957, -5.714286) " points="12.1462424 4.99661476 6.43195667 -0.717670955 0.717670955 4.99661476 2.38320921 6.66215301 5.25389303 3.77792823 5.25389303 12.1462424 7.61002031 12.1462424 7.61002031 3.77792823 10.4807041 6.66215301"></polygon></g></g></g></g></svg>Back to Main Menu</button>
          <div class="submenus-wrapper-inner">
        <ul class="submenu">
                              <li  class="menu-item">
              
 
  <a href="https://today.usc.edu/news/"  class="">Home</a>



  
            </li>
                                        <li  class="menu-item">
              
   <div class="link-arrow-wrapper">
    <button class="expand" type="button" aria-expanded="false" aria-label="Display Sub Menu for Sections" aria-controls="sections61">
      Sections
      <span class="icon">
        <svg width="7px" height="11px" viewBox="0 0 7 11" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><g stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"><g transform="translate(-230.000000, -41.000000)" fill="#333333"><g><g><g transform="translate(0.000000, 24.000000)"><g transform="translate(220.000000, 10.000000)"><polygon transform="translate(13.137755, 12.500000) scale(-1, 1) rotate(90.000000) translate(-13.137755, -12.500000) " points="13.1377551 13.8010204 17.5765306 9.3622449 18.494898 10.2806122 13.1377551 15.6377551 7.78061224 10.2806122 8.69897959 9.3622449"></polygon></g></g></g></g></g></g></svg>
      </span>
    </button>
  </div>



  
  <div class="submenus-wrapper" id="sections61">
    <button class="back-to-prev"><svg width="13px" height="12px" viewBox="0 0 13 12" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><g stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"><g transform="translate(-64.000000, -167.000000)" fill="#FFFFFF"><g transform="translate(64.000000, 160.000000)"><g transform="translate(0.000000, 7.000000)"><polygon transform="translate(6.431957, 5.714286) scale(-1, 1) rotate(-270.000000) translate(-6.431957, -5.714286) " points="12.1462424 4.99661476 6.43195667 -0.717670955 0.717670955 4.99661476 2.38320921 6.66215301 5.25389303 3.77792823 5.25389303 12.1462424 7.61002031 12.1462424 7.61002031 3.77792823 10.4807041 6.66215301"></polygon></g></g></g></g></svg>Back to Sections</button>
          <div class="submenus-wrapper-inner">
        <ul class="submenu">
                                    <li  class="menu-item">
              <a  href="https://today.usc.edu/category/arts/" class="">Arts</a>
                              </li>
                                                <li  class="menu-item">
              <a  href="https://today.usc.edu/category/athletics/" class="">Athletics</a>
                              </li>
                                                <li  class="menu-item">
              <a  href="https://today.usc.edu/category/business/" class="">Business</a>
                              </li>
                                                <li  class="menu-item">
              <a  href="https://today.usc.edu/category/health/" class="">Health</a>
                              </li>
                                                <li  class="menu-item">
              <a  href="https://today.usc.edu/category/humanities/" class="">Humanities</a>
                              </li>
                                                <li  class="menu-item">
              <a  href="https://today.usc.edu/category/policy-law/" class="">Policy/Law</a>
                              </li>
                                                <li  class="menu-item">
              <a  href="https://today.usc.edu/category/science-technology/" class="">Science/Technology</a>
                              </li>
                                                <li  class="menu-item">
              <a  href="https://today.usc.edu/category/social-impact/" class="">Social Impact</a>
                              </li>
                                                <li  class="menu-item">
              <a  href="https://today.usc.edu/category/student-life/" class="">Student Life</a>
                              </li>
                                                <li  class="menu-item">
              <a  href="https://today.usc.edu/category/university/" class="">University</a>
                              </li>
                              </ul>
      </div>
      </div>
            </li>
                                        <li  class="menu-item">
              
 
  <a href="https://today.usc.edu/news/contact/"  class="">Contact</a>



  
            </li>
                                        <li  class="menu-item">
              
 
  <a href="https://today.usc.edu/?s="  class="">Search</a>



  
            </li>
                          </ul>
      </div>
      </div>
              </li>
                                                <li  class="menu-item">
               
  <div class="link-arrow-wrapper">
    <button class="expand" type="button" aria-expanded="false" aria-label="Display Sub Menu for USC Trojan Family Magazine" aria-controls="usc-trojan-family-magazine83">
      USC Trojan Family Magazine
      <span class="icon">
        <svg width="7px" height="11px" viewBox="0 0 7 11" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><g stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"><g transform="translate(-230.000000, -41.000000)" fill="#333333"><g><g><g transform="translate(0.000000, 24.000000)"><g transform="translate(220.000000, 10.000000)"><polygon transform="translate(13.137755, 12.500000) scale(-1, 1) rotate(90.000000) translate(-13.137755, -12.500000) " points="13.1377551 13.8010204 17.5765306 9.3622449 18.494898 10.2806122 13.1377551 15.6377551 7.78061224 10.2806122 8.69897959 9.3622449"></polygon></g></g></g></g></g></g></svg>
      </span>
    </button>
  </div>


  
  <div class="submenus-wrapper" id="usc-trojan-family-magazine83">
    <button class="back-to-prev"><svg width="13px" height="12px" viewBox="0 0 13 12" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><g stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"><g transform="translate(-64.000000, -167.000000)" fill="#FFFFFF"><g transform="translate(64.000000, 160.000000)"><g transform="translate(0.000000, 7.000000)"><polygon transform="translate(6.431957, 5.714286) scale(-1, 1) rotate(-270.000000) translate(-6.431957, -5.714286) " points="12.1462424 4.99661476 6.43195667 -0.717670955 0.717670955 4.99661476 2.38320921 6.66215301 5.25389303 3.77792823 5.25389303 12.1462424 7.61002031 12.1462424 7.61002031 3.77792823 10.4807041 6.66215301"></polygon></g></g></g></g></svg>Back to Main Menu</button>
          <div class="submenus-wrapper-inner">
        <ul class="submenu">
                              <li  class="menu-item">
              
 
  <a href="https://today.usc.edu/trojan-family-magazine/"  class="">Home</a>



  
            </li>
                                        <li  class="menu-item">
              
   <div class="link-arrow-wrapper">
    <button class="expand" type="button" aria-expanded="false" aria-label="Display Sub Menu for Features & News" aria-controls="features-&-news44">
      Features & News
      <span class="icon">
        <svg width="7px" height="11px" viewBox="0 0 7 11" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><g stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"><g transform="translate(-230.000000, -41.000000)" fill="#333333"><g><g><g transform="translate(0.000000, 24.000000)"><g transform="translate(220.000000, 10.000000)"><polygon transform="translate(13.137755, 12.500000) scale(-1, 1) rotate(90.000000) translate(-13.137755, -12.500000) " points="13.1377551 13.8010204 17.5765306 9.3622449 18.494898 10.2806122 13.1377551 15.6377551 7.78061224 10.2806122 8.69897959 9.3622449"></polygon></g></g></g></g></g></g></svg>
      </span>
    </button>
  </div>



  
  <div class="submenus-wrapper" id="features-&-news44">
    <button class="back-to-prev"><svg width="13px" height="12px" viewBox="0 0 13 12" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><g stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"><g transform="translate(-64.000000, -167.000000)" fill="#FFFFFF"><g transform="translate(64.000000, 160.000000)"><g transform="translate(0.000000, 7.000000)"><polygon transform="translate(6.431957, 5.714286) scale(-1, 1) rotate(-270.000000) translate(-6.431957, -5.714286) " points="12.1462424 4.99661476 6.43195667 -0.717670955 0.717670955 4.99661476 2.38320921 6.66215301 5.25389303 3.77792823 5.25389303 12.1462424 7.61002031 12.1462424 7.61002031 3.77792823 10.4807041 6.66215301"></polygon></g></g></g></g></svg>Back to Features & News</button>
          <div class="submenus-wrapper-inner">
        <ul class="submenu">
                                    <li  class="menu-item">
              <a  href="https://today.usc.edu/trojan-family-magazine/tfm-small-bites/" class="">Small Bites</a>
                              </li>
                                                <li  class="menu-item">
              <a  href="https://today.usc.edu/trojan-family-magazine/class-notes-2/" class="">Class Notes</a>
                              </li>
                              </ul>
      </div>
      </div>
            </li>
                                        <li  class="menu-item">
              
 
  <a href="https://today.usc.edu/trojan-family-magazine/magazine-issues/"  class="">Archive</a>



  
            </li>
                                        <li  class="menu-item">
              
 
  <a href="https://usc.qualtrics.com/jfe/form/SV_ezLBO5KQUQMoGzj"  class="">Submit</a>



  
            </li>
                                        <li  class="menu-item">
              
 
  <a href="https://today.usc.edu/?s="  class="">Search</a>



  
            </li>
                          </ul>
      </div>
      </div>
              </li>
                              </ul>
      </nav>
    </div>
  
    
    






<div
  class="cc--component-container cc--secondary-menu gold"

  
  
  
  
  
  
  >
  <div class="c--component c--secondary-menu"
    
      >

    
  

  </div></div>

    
    






<div
  class="cc--component-container cc--header-image "

  
  
  
  
  
  
  >
  <div class="c--component c--header-image"
    
      >

    
      <div class="image-container">
          
<div class="f--field f--image">

    
    
    
    
    
    
              
      <img
                            data-src="https://today.usc.edu/wp-content/uploads/2023/09/41877449584_2846dd729a_o-768x768.jpg"
          data-srcset="https://today.usc.edu/wp-content/uploads/2023/09/41877449584_2846dd729a_o-1280x1280.jpg 1280w,https://today.usc.edu/wp-content/uploads/2023/09/41877449584_2846dd729a_o-768x768.jpg 768w"          data-sizes="(min-width:1024px) 50vw, (min-width:768px) 100vw, 100vw"          class="lazyload"
        
        alt="Beauty shot of fountain and Hahn Plaza at USC"
        
                                      />

    
    
  
  

</div>
    </div>
  
  

  </div></div>


  </div></div>

          <div class="supplementary-menus">

              
    






<div
  class="cc--component-container cc--utility-menu gold"

  
  
  
  
  
  
  >
  <div class="c--component c--utility-menu"
    
      >

    
      <div class="menu-container">
      <nav aria-label="Utility Menu">
        <ul class="m--menu">
                      <li class="menu-item">

                              <a href="https://today.usc.edu/about-this-site/" >About This Site</a>
              
            </li>
                      <li class="menu-item">

                              <a href="https://today.usc.edu/contact-us/" >Contact Us</a>
              
            </li>
                  </ul>
      </nav>
    </div>

  

  </div></div>

              
    






<div
  class="cc--component-container cc--audiences-menu gold"

  
  
  
  
  
  
  >
  <div class="c--component c--audiences-menu"
    
      >

    
      <div class="menu-container">
      <nav>
        <ul class="m--menu" aria-label="Audience Menu">
                      <li class="menu-item">
                              <a href="https://today.usc.edu/pressroom/" >Pressroom</a>
                          </li>
                      <li class="menu-item">
                              <a href="https://today.usc.edu/pressroom/find-an-expert/" >Find an Expert</a>
                          </li>
                      <li class="menu-item">
                              <a href="https://today.usc.edu/pressroom/media-contacts/" >Media Contacts</a>
                          </li>
                      <li class="menu-item">
                              <a href="https://today.usc.edu/campus-access-for-media/" >Campus Access for Media</a>
                          </li>
                  </ul>
      </nav>
    </div>
  

  </div></div>

                      </div>
        </div>
      </div>

        
    






<div
  class="cc--component-container cc--search-form gold"

   id="search-container"
  
   role="dialog"
  
  
  
  >
  <div class="c--component c--search-form"
    
      >

    
      
<div class="f--field f--section-title">

    
  <h2>What are you looking for?</h2>


</div>

  <form autocomplete="on" name="search-form" method="get" action="/?s=" >
  
      
      <label class="visually-hidden" for="search">Search:</label>
      <div class="input-wrapper">
        <div class="fi--form-item fi--search">
          <input
            type="text"
            id="search"
            maxlength="150"
            class="form-autocomplete form-text ui-autocomplete-input"
            name="s"
            autocomplete="off"
            data-drupal-selector="edit-search-2"
            placeholder="Search USC"
            size="150" >
        </div>
      </div>

      <div class="fi--form-item fi--form-item-submit ">
  
  <button>
    Search

      </button>

</div>

    </form>

  

  </div></div>

    </div>

    
                          
    






<div
  class="cc--component-container cc--horizontal-strip gold"

  
  
  
  
  
  
  >
  <div class="c--component c--horizontal-strip"
    
      >

    
      <div class="menu-container">
      <div class="strip-wrapper">
        <button class="horizontal-strip-toggle" aria-label="Expand Horizontal Menu">
          <svg class="open-icon" height="7" viewBox="0 0 11 7" width="11" xmlns="http://www.w3.org/2000/svg"><path d="m3.1377551 6.65816327 4.43877551-4.43877551.91836735.91836734-5.35714286 5.35714286-5.35714286-5.35714286.91836735-.91836734z" fill="#fff" fill-rule="evenodd" transform="matrix(-1 0 0 1 8.637755 -2.219388)"/></svg>
          <svg class="close-icon" height="18" viewBox="0 0 18 18" width="18" xmlns="http://www.w3.org/2000/svg"><path d="m25.4719587 9.08885563 1.4142136 1.41421357-7.4843167 7.4847864 7.4843167 7.4841031-1.4142136 1.4142136-7.4841031-7.4843167-7.4847864 7.4843167-1.41421357-1.4142136 7.48499997-7.4841031-7.48499997-7.4847864 1.41421357-1.41421357 7.4847864 7.48499997z" fill="#fff" fill-rule="evenodd" transform="translate(-9 -9)"/></svg>
        </button>
        <ul class="breadcrumb-menu">
                                                            <li class="menu-item">
                USC News
              </li>
            
                        <li class="menu-item">
        <span>Home</span>
      </li>
    
    
      
          
    
      
    
      
    
      
    
      
    
      
    
      
    
      
    
      
    
      
    
  
      
    
      
    
  
                  </ul>
      </div>
      <nav class="horizontal-strip-nav">
        <ul class="m--menu">
                                                                                                                  <li class="menu-item menu-item--active-trail">
                      
 
  <a href="https://today.usc.edu/news/"  class="">Home</a>




  
                    </li>
                                      <li class="menu-item ">
                      
   <div class="link-arrow-wrapper">
    <button class="expand" type="button" aria-expanded="false" aria-label="Display Sub Menu for Sections" aria-controls="hor-nav-sections29">
      Sections
      <span class="icon">
        <svg width="7px" height="11px" viewBox="0 0 7 11" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><g stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"><g transform="translate(-230.000000, -41.000000)" fill="#333333"><g><g><g transform="translate(0.000000, 24.000000)"><g transform="translate(220.000000, 10.000000)"><polygon transform="translate(13.137755, 12.500000) scale(-1, 1) rotate(90.000000) translate(-13.137755, -12.500000) " points="13.1377551 13.8010204 17.5765306 9.3622449 18.494898 10.2806122 13.1377551 15.6377551 7.78061224 10.2806122 8.69897959 9.3622449"></polygon></g></g></g></g></g></g></svg>
      </span>
    </button>
  </div>



  
  <div class="submenus-wrapper" id="hor-nav-sections29">
    <button class="back-to-prev"><svg height="12" viewBox="0 0 20 12" width="20" xmlns="http://www.w3.org/2000/svg"><path d="m22.0243902 12 5.9756098 5.9756098-1 .9983902v.0173244l-.017-.0003244-4.9586098 4.9602195-1.0243902-1.0243902 3.935-3.9358293-16.935.0003244v-2l16.967-.0003244-3.967-3.9666098z" fill="#fff" fill-rule="evenodd" transform="matrix(-1 0 0 -1 28 23.95122)"/></svg>Back to Sections Menu</button>
          <div class="submenus-wrapper-inner">
        <ul class="submenu">
                  <li class="menu-item ">
            
 
  <a href="https://today.usc.edu/category/arts/"  class="">Arts</a>




  
          </li>
                  <li class="menu-item ">
            
 
  <a href="https://today.usc.edu/category/athletics/"  class="">Athletics</a>




  
          </li>
                  <li class="menu-item ">
            
 
  <a href="https://today.usc.edu/category/business/"  class="">Business</a>




  
          </li>
                  <li class="menu-item ">
            
 
  <a href="https://today.usc.edu/category/health/"  class="">Health</a>




  
          </li>
                  <li class="menu-item ">
            
 
  <a href="https://today.usc.edu/category/humanities/"  class="">Humanities</a>




  
          </li>
                  <li class="menu-item ">
            
 
  <a href="https://today.usc.edu/category/policy-law/"  class="">Policy/Law</a>




  
          </li>
                  <li class="menu-item ">
            
 
  <a href="https://today.usc.edu/category/science-technology/"  class="">Science/Technology</a>




  
          </li>
                  <li class="menu-item ">
            
 
  <a href="https://today.usc.edu/category/social-impact/"  class="">Social Impact</a>




  
          </li>
                  <li class="menu-item ">
            
 
  <a href="https://today.usc.edu/category/student-life/"  class="">Student Life</a>




  
          </li>
                  <li class="menu-item ">
            
 
  <a href="https://today.usc.edu/category/university/"  class="">University</a>




  
          </li>
                </ul>
      </div>
      </div>
                    </li>
                                      <li class="menu-item ">
                      
 
  <a href="https://today.usc.edu/news/contact/"  class="">Contact</a>




  
                    </li>
                                      <li class="menu-item ">
                      
 
  <a href="https://today.usc.edu/?s="  class="">Search</a>




  
                    </li>
                                                                              </ul>
      </nav>
    </div>

  

  </div></div>
                  
  </div>


  </div></div>

				
			</div>
		
		
	</header>

		<main id="main-content">
			<div class="l--content">
				<div class="content">
					
	<section class="tc--template-container tc--content-page">

		<div class="t--template t--content-page">
						
			<div class="content-main">
				<div class="lc--layout-container lc--full">
					<div class="l--layout l--full">
						<div class="lr--layout-region lr--main">
								<div class="content-wrapper">
		<article class="post-type-page" id="post-1271">
			<div class="article-content">
				<div class="article-body">
					  
    






<div
  class="cc--component-container cc--hidden-page-title "

  
  
  
  
  
  
  >
  <div class="c--component c--hidden-page-title"
    
      >

              
<div class="f--field f--page-title visually-hidden">

    
  <h1>USC News</h1>


</div>
  
  </div></div>



  

    
      
    
      
    
      
    
      
  
    






<div
  class="cc--component-container cc--featured-articles "

  
  
  
  
  
  
  >
  <div class="c--component c--featured-articles"
    
      >

    
  <div class="header-container">
              
<div class="f--field f--section-title">

    
  <h2>Top Stories</h2>


</div>
    
              
<div class="f--field f--link">

    
    
  
<a class="link "  href=https://today.usc.edu/tag/usc-news/  aria-label="Read more about ALL USC NEWS STORIES">ALL USC NEWS STORIES</a>


</div>
      </div>

  <div class="flex-container">

          <div class="featured-card">
                            
              
<div class="f--field f--image">

    
        <a href="https://dornsife.usc.edu/news/stories/computational-linguistics-student-finds-love-of-language-and-ai-safety-at-usc-dornsife/"  target="_blank"   aria-label="Love of language and community — and AI safety — fuel graduate’s USC experience" >
  
    
    
    
    
              
      <img
                            data-src="https://today.usc.edu/wp-content/uploads/2026/05/kyle-ng-grad-profile-central-768x432.jpg"
          data-srcset="https://today.usc.edu/wp-content/uploads/2026/05/kyle-ng-grad-profile-central-1920x1080.jpg 1920w,https://today.usc.edu/wp-content/uploads/2026/05/kyle-ng-grad-profile-central-1280x720.jpg 1280w,https://today.usc.edu/wp-content/uploads/2026/05/kyle-ng-grad-profile-central-768x432.jpg 768w"          data-sizes="(min-width:1200px) 75vw, (min-width:768px) 83vw, 100vw"          class="lazyload"
        
        alt="Kyle Ng"
        
                                      />

    
    
  
      </a>
  

</div>
        
                      
<div class="f--field f--eyebrow">

    
  <span>University</span>


</div>
        
                      
<div class="f--field f--cta-title">

    
  <h3>
          <a href="https://dornsife.usc.edu/news/stories/computational-linguistics-student-finds-love-of-language-and-ai-safety-at-usc-dornsife/"  target="_blank"  >Love of language and community — and AI safety — fuel graduate’s USC experience</a>
      </h3>


</div>
        
                      
<div class="f--field f--description">

    
  <p><strong>COMMENCEMENT:</strong> USC Dornsife computational linguistics major Kyle Ng explores ways to make large language models safer while also helping his fellow students feel at home on campus.</p>



</div>
        
                      
<div class="f--field f--eyebrow date">

    
  <span>May 8, 2026</span>


</div>
        
      </div>
    
          <div class="cards-list">
                                                <div class="article-card">
                                              
                    
<div class="f--field f--image">

    
        <a href="https://priceschool.usc.edu/news/sunstone-management-economic-development-challenge/"  target="_blank"   aria-label="USC Price students’ work with Riverside city officials win $5,000 at Sunstone Economic Development Challenge" >
  
    
    
    
    
              
      <img
                            data-src="https://today.usc.edu/wp-content/uploads/2026/05/Sunstone-Management-economic-development-challenge-2026_uscn_3750-320x180.jpg"
          data-srcset="https://today.usc.edu/wp-content/uploads/2026/05/Sunstone-Management-economic-development-challenge-2026_uscn_3750-768x432.jpg 768w,https://today.usc.edu/wp-content/uploads/2026/05/Sunstone-Management-economic-development-challenge-2026_uscn_3750-320x180.jpg 320w"          data-sizes="(min-width:1024px) 23vw, (min-width:768px) 39vw, 83vw"          class="lazyload"
        
        alt="Winning team: Hilda Aguilar-Jurado, Vanessa Bautista, Oscar Cisco and Montserrat Pineda"
        
                                      />

    
    
  
      </a>
  

</div>
              
              <div class="text-container">

                                      
<div class="f--field f--eyebrow">

    
  <span>Social Impact</span>


</div>
                
                                      
<div class="f--field f--cta-title">

    
  <h3>
          <a href="https://priceschool.usc.edu/news/sunstone-management-economic-development-challenge/"  target="_blank"  >USC Price students’ work with Riverside city officials win $5,000 at Sunstone Economic Development Challenge</a>
      </h3>


</div>
                
                                      
<div class="f--field f--description">

    
  <p>The winning team worked to address a growing misalignment between the skills of its workforce and the needs of local employers.</p>



</div>
                
                                      
<div class="f--field f--eyebrow date">

    
  <span>May 8, 2026</span>


</div>
                              </div>

            </div>
                                        <div class="article-card">
                                              
                    
<div class="f--field f--image">

    
        <a href="https://www.keckmedicine.org/blog/what-should-patients-look-for-in-an-organ-transplant-center/"  target="_blank"   aria-label="What should patients look for in an organ transplant center?" >
  
    
    
    
    
              
      <img
                            data-src="https://today.usc.edu/wp-content/uploads/2026/05/AdobeStock_705577663_uscn_4697-320x180.jpg"
          data-srcset="https://today.usc.edu/wp-content/uploads/2026/05/AdobeStock_705577663_uscn_4697-1280x720.jpg 1280w,https://today.usc.edu/wp-content/uploads/2026/05/AdobeStock_705577663_uscn_4697-768x432.jpg 768w,https://today.usc.edu/wp-content/uploads/2026/05/AdobeStock_705577663_uscn_4697-320x180.jpg 320w"          data-sizes="(min-width:1024px) 23vw, (min-width:768px) 39vw, 83vw"          class="lazyload"
        
        alt="Person using laptop"
        
                                      />

    
    
  
      </a>
  

</div>
              
              <div class="text-container">

                                      
<div class="f--field f--eyebrow">

    
  <span>Health</span>


</div>
                
                                      
<div class="f--field f--cta-title">

    
  <h3>
          <a href="https://www.keckmedicine.org/blog/what-should-patients-look-for-in-an-organ-transplant-center/"  target="_blank"  >What should patients look for in an organ transplant center?</a>
      </h3>


</div>
                
                                      
<div class="f--field f--description">

    
  <p>Mark Sturdevant, a transplant surgeon with the USC Transplant Institute, lists key differentiators.</p>



</div>
                
                                      
<div class="f--field f--eyebrow date">

    
  <span>May 8, 2026</span>


</div>
                              </div>

            </div>
                                        <div class="article-card">
                                              
                    
<div class="f--field f--image">

    
        <a href="https://gero.usc.edu/2026/05/05/msnhl-veteran-2026/"  target="_blank"   aria-label="Veteran shares journey to USC master’s in nutrition, healthspan and longevity" >
  
    
    
    
    
              
      <img
                            data-src="https://today.usc.edu/wp-content/uploads/2026/05/20250808_123405_0000_uscn_0134-320x180.jpg"
          data-srcset="https://today.usc.edu/wp-content/uploads/2026/05/20250808_123405_0000_uscn_0134-768x432.jpg 768w,https://today.usc.edu/wp-content/uploads/2026/05/20250808_123405_0000_uscn_0134-320x180.jpg 320w"          data-sizes="(min-width:1024px) 23vw, (min-width:768px) 39vw, 83vw"          class="lazyload"
        
        alt="Faith Huss"
        
                                      />

    
    
  
      </a>
  

</div>
              
              <div class="text-container">

                                      
<div class="f--field f--eyebrow">

    
  <span>University</span>


</div>
                
                                      
<div class="f--field f--cta-title">

    
  <h3>
          <a href="https://gero.usc.edu/2026/05/05/msnhl-veteran-2026/"  target="_blank"  >Veteran shares journey to USC master’s in nutrition, healthspan and longevity</a>
      </h3>


</div>
                
                                      
<div class="f--field f--description">

    
  <p><strong>COMMENCEMENT:</strong> Faith Huss, who served in the U.S. Army for 10 years prior to attending USC, will use her nutrition and dietetics expertise to serve patients at the Veterans Affairs Hospital in Long Beach.</p>



</div>
                
                                      
<div class="f--field f--eyebrow date">

    
  <span>May 7, 2026</span>


</div>
                              </div>

            </div>
                        </div>
    
  </div>




  </div></div>




<div class="cc--component-container cc--countdown">
  <div class="c--component c--countdown">
    <span id="countdown_date_time" style="display:none;">May 14, 2026 18:30:00 PDT</span>
          <div class="header-container">
        <div class="f--field f--section-title">
          <h2>Countdown to Commencement</h2>
        </div>
      </div>
    
    <div class="countdown-container">

      <div class="countdown-item" id="days">
        <h3></h3>
        <span>Days</span>
      </div>

      <div class="countdown-item" id="hours">
        <h3></h3>
        <span>Hours</span>
      </div>

      <div class="countdown-item" id="minutes">
        <h3></h3>
        <span>Minutes</span>
      </div>

      <div class="countdown-item" id="seconds">
        <h3></h3>
        <span>Seconds</span>
      </div>
    </div>
  </div>
</div>


  
    






<div
  class="cc--component-container cc--spacer "

  
  
  
  
  
  
  >
  <div class="c--component c--spacer"
    
      >

    

  </div></div>




  
  
    






<div
  class="cc--component-container cc--full-width-image-and-text text-left"

  
  
  
  
  
  
  >
  <div class="c--component c--full-width-image-and-text"
    
      >

    
      <div class="image-container">
          
<div class="f--field f--image">

    
    
    
    
    
    
              
      <img
                            data-src="https://today.usc.edu/wp-content/uploads/2026/05/Stevens_Photo-Illustration-768x432.jpg"
          data-srcset="https://today.usc.edu/wp-content/uploads/2026/05/Stevens_Photo-Illustration-1920x1080.jpg 1920w,https://today.usc.edu/wp-content/uploads/2026/05/Stevens_Photo-Illustration-1280x720.jpg 1280w,https://today.usc.edu/wp-content/uploads/2026/05/Stevens_Photo-Illustration-768x432.jpg 768w"          data-sizes="(min-width:1200px) 75vw, (min-width:768px) 83vw, 100vw"          class="lazyload"
        
        alt="Mark and Mary Stevens"
        
                                      />

    
    
  
  

</div>
    </div>
  
  <div class="text-container">

              
<div class="f--field f--section-title">

    
  <h2>Mark and Mary Stevens Give $200 Million to Power AI Research Across USC</h2>


</div>
    
    <div class="col">
                  
<div class="f--field f--description">

    
  <p>The university will name the USC Mark and Mary Stevens School of Computing and Artificial Intelligence to honor their investment in research and innovation at the intersections of AI and health sciences, business, security and the arts.</p>



</div>
      
              <div class="links-container">
                          
<div class="f--field f--link">

    
    
  
<a class="link "  href=https://today.usc.edu/mark-and-mary-stevens-give-200-million-to-power-ai-research-across-usc/ target="_blank" aria-label="Read more about Learn more">Learn more</a>


</div>
                  </div>
          </div>
  </div>


  </div></div>


  
    






<div
  class="cc--component-container cc--spacer "

  
  
  
  
  
  
  >
  <div class="c--component c--spacer"
    
      >

    

  </div></div>



            
  
    






<div
  class="cc--component-container cc--link-list "

  
  
  
  
  
  
  >
  <div class="c--component c--link-list"
    
      >

    
      <div class="header-container">

                  
<div class="f--field f--section-title">

    
  <h2>USC in the News</h2>


</div>
                </div>
  
      <div class="links-container">

                  
      
        
          <div class="column">
              <ul>
                                                            <li>
                              
<div class="f--field f--link">

    
    
  
<a class="link "  href=https://www.npr.org/2026/04/06/nx-s1-5745209/shingles-symptoms-vaccine-treatment target="_blank" aria-label="Read more about NPR ◆ Shingles can hit younger than you think, and there are more reasons to get vaccinated">NPR ◆ Shingles can hit younger than you think, and there are more reasons to get vaccinated</a>


</div>

                                                </li>
                                                                                <li>
                              
<div class="f--field f--link">

    
    
  
<a class="link "  href=https://www.cbsnews.com/losangeles/news/nasa-usc-robotics-team-up-to-strengthen-space-exploration-program/ target="_blank" aria-label="Read more about CBS NEWS ◆ NASA, USC robotics team for space exploration program">CBS NEWS ◆ NASA, USC robotics team for space exploration program</a>


</div>

                                                </li>
                                                                                <li>
                              
<div class="f--field f--link">

    
    
  
<a class="link "  href=https://www.sfweekly.com/lifestyles/health/is-even-the-occasional-night-out-bad-for-your-liver/article_4dc2e84e-3c96-527e-96cb-3b1166c53c40.html  aria-label="Read more about SF WEEKLY ◆ Is even the occasional night out bad for your liver?">SF WEEKLY ◆ Is even the occasional night out bad for your liver?</a>


</div>

                                                </li>
                                                    </ul>
          </div>

        
        
      
        
          <div class="column">
              <ul>
                                                            <li>
                              
<div class="f--field f--link">

    
    
  
<a class="link "  href=https://www.sandiegouniontribune.com/2026/03/11/ucsd-study-detects-dementia-risk-decades-before-symptoms-appear/ target="_blank" aria-label="Read more about THE SAN DIEGO UNION-TRIBUNE ◆ Study detects dementia risk decades before symptoms appear">THE SAN DIEGO UNION-TRIBUNE ◆ Study detects dementia risk decades before symptoms appear</a>


</div>

                                                </li>
                                                                                <li>
                              
<div class="f--field f--link">

    
    
  
<a class="link "  href=https://www.forbes.com/councils/forbesbusinesscouncil/2026/02/25/the-communication-gap-many-leaders-dont-see-and-how-to-close-it/ target="_blank" aria-label="Read more about FORBES ◆ The workplace communication gap many leaders don’t see is driving employees away">FORBES ◆ The workplace communication gap many leaders don’t see is driving employees away</a>


</div>

                                                </li>
                                                                                <li>
                              
<div class="f--field f--link">

    
    
  
<a class="link "  href=https://time.com/7379213/recycling-symbol-history-plastic-inc-book-excerpt/ target="_blank" aria-label="Read more about TIME ◆ What the Trojan who created the recycling symbol thinks today">TIME ◆ What the Trojan who created the recycling symbol thinks today</a>


</div>

                                                </li>
                                                    </ul>
          </div>

        
        
      

    </div>
  

  </div></div>


  
    






<div
  class="cc--component-container cc--spacer "

  
  
  
  
  
  
  >
  <div class="c--component c--spacer"
    
      >

    

  </div></div>




  
  
    






<div
  class="cc--component-container cc--full-width-image-and-text "

  
  
  
  
  
  
  >
  <div class="c--component c--full-width-image-and-text"
    
      >

    
      <div class="image-container">
          
<div class="f--field f--image">

    
    
    
    
    
    
              
      <img
                            data-src="https://today.usc.edu/wp-content/uploads/2025/07/iStock-1400699950-768x432.jpg"
          data-srcset="https://today.usc.edu/wp-content/uploads/2025/07/iStock-1400699950-1920x1080.jpg 1920w,https://today.usc.edu/wp-content/uploads/2025/07/iStock-1400699950-1280x720.jpg 1280w,https://today.usc.edu/wp-content/uploads/2025/07/iStock-1400699950-768x432.jpg 768w"          data-sizes="(min-width:1200px) 75vw, (min-width:768px) 83vw, 100vw"          class="lazyload"
        
        alt="Tree-lined street"
        
                                      />

    
    
  
  

</div>
    </div>
  
  <div class="text-container">

              
<div class="f--field f--section-title">

    
  <h2>Bringing Shade to Los Angeles Before The Olympics</h2>


</div>
    
    <div class="col">
                  
<div class="f--field f--description">

    
  <p>ShadeLA, a new USC-led initiative, unites civic, academic and community partners to expand urban tree canopy and shade infrastructure — building lasting heat resilience for Angelenos well beyond 2028.</p>



</div>
      
              <div class="links-container">
                          
<div class="f--field f--link">

    
    
  
<a class="link "  href=https://today.usc.edu/usc-teams-up-with-shadela-coalition-to-cool-los-angeles-ahead-of-2028/ target="_blank" aria-label="Read more about Story and photos in USC News">Story and photos in USC News</a>


</div>
                  </div>
          </div>
  </div>


  </div></div>


  
    






<div
  class="cc--component-container cc--spacer "

  
  
  
  
  
  
  >
  <div class="c--component c--spacer"
    
      >

    

  </div></div>



      
        
        
        
  
  
    






<div
  class="cc--component-container cc--card-grid "

  
  
  
  
  
  
  >
  <div class="c--component c--card-grid"
    
      >

    
      <div class="header-container">
                  
<div class="f--field f--section-title">

    
  <h2>Trending</h2>


</div>
      
                  
<div class="f--field f--link">

    
    
  
<a class="link "  href=https://today.usc.edu/tag/usc-news/  aria-label="Read more about All USC News Stories">All USC News Stories</a>


</div>
      
    </div>
    
      <div class="cards-container">
      
        <div class="card">

          
                        
                
<div class="f--field f--image">

    
        <a href="https://today.usc.edu/the-water-wars-of-the-future-are-here-today/"   aria-label="The water wars of the future are here today" >
  
    
    
    
    
              
      <img
                            data-src="https://today.usc.edu/wp-content/uploads/2023/02/iStock-1060699886_WEB-320x180.jpg"
          data-srcset="https://today.usc.edu/wp-content/uploads/2023/02/iStock-1060699886_WEB-1280x720.jpg 1280w,https://today.usc.edu/wp-content/uploads/2023/02/iStock-1060699886_WEB-768x432.jpg 768w,https://today.usc.edu/wp-content/uploads/2023/02/iStock-1060699886_WEB-320x180.jpg 320w"          data-sizes="(min-width:1024px) 23vw, (min-width:768px) 39vw, 83vw"          class="lazyload"
        
        alt="Colorado River"
        
                                      />

    
    
  
      </a>
  

</div>
          
          <div class="text-container">
                              
<div class="f--field f--eyebrow">

    
  <span>Policy/Law</span>


</div>
            
                            <h3>
                                      <a href="https://today.usc.edu/the-water-wars-of-the-future-are-here-today/" >
                  
                    The water wars of the future are here today

                                      </a>
                                  </h3>
                      </div>

        </div>
      
        <div class="card">

          
                        
                
<div class="f--field f--image">

    
        <a href="https://dornsife.usc.edu/news/stories/3393/slaves-escape-via-underground-railroad-to-mexico/"  target="_blank"   aria-label="USC historian uncovers the Underground Railroad that ran to Mexico" >
  
    
    
    
    
              
      <img
                            data-src="https://today.usc.edu/wp-content/uploads/2021/02/south-to-freedom-web-320x180.jpg"
          data-srcset="https://today.usc.edu/wp-content/uploads/2021/02/south-to-freedom-web-1280x720.jpg 1280w,https://today.usc.edu/wp-content/uploads/2021/02/south-to-freedom-web-768x432.jpg 768w,https://today.usc.edu/wp-content/uploads/2021/02/south-to-freedom-web-320x180.jpg 320w"          data-sizes="(min-width:1024px) 23vw, (min-width:768px) 39vw, 83vw"          class="lazyload"
        
        alt="family riding on a horse illustration"
        
                                      />

    
    
  
      </a>
  

</div>
          
          <div class="text-container">
                              
<div class="f--field f--eyebrow">

    
  <span>Humanities</span>


</div>
            
                            <h3>
                                      <a href="https://dornsife.usc.edu/news/stories/3393/slaves-escape-via-underground-railroad-to-mexico/" target="_blank">
                  
                    USC historian uncovers the Underground Railroad that ran to Mexico

                                      </a>
                                  </h3>
                      </div>

        </div>
      
        <div class="card">

          
                        
                
<div class="f--field f--image">

    
        <a href="https://today.usc.edu/what-social-rejection-teaches-your-brain/"   aria-label="What social rejection teaches your brain" >
  
    
    
    
    
              
      <img
                            data-src="https://today.usc.edu/wp-content/uploads/2024/12/iStock-902027526-320x180.jpg"
          data-srcset="https://today.usc.edu/wp-content/uploads/2024/12/iStock-902027526-1280x720.jpg 1280w,https://today.usc.edu/wp-content/uploads/2024/12/iStock-902027526-768x432.jpg 768w,https://today.usc.edu/wp-content/uploads/2024/12/iStock-902027526-320x180.jpg 320w"          data-sizes="(min-width:1024px) 23vw, (min-width:768px) 39vw, 83vw"          class="lazyload"
        
        alt="Photo illustration: bird isolated from other birds"
        
                                      />

    
    
  
      </a>
  

</div>
          
          <div class="text-container">
                              
<div class="f--field f--eyebrow">

    
  <span>Science/Technology</span>


</div>
            
                            <h3>
                                      <a href="https://today.usc.edu/what-social-rejection-teaches-your-brain/" >
                  
                    What social rejection teaches your brain

                                      </a>
                                  </h3>
                      </div>

        </div>
      
        <div class="card">

          
                        
                
<div class="f--field f--image">

    
        <a href="https://today.usc.edu/using-ai-to-predict-wildfires/"   aria-label="USC scientists use AI to predict a wildfire’s next move" >
  
    
    
    
    
              
      <img
                            data-src="https://today.usc.edu/wp-content/uploads/2024/07/iStock-1441834978-320x180.jpg"
          data-srcset="https://today.usc.edu/wp-content/uploads/2024/07/iStock-1441834978-1280x720.jpg 1280w,https://today.usc.edu/wp-content/uploads/2024/07/iStock-1441834978-768x432.jpg 768w,https://today.usc.edu/wp-content/uploads/2024/07/iStock-1441834978-320x180.jpg 320w"          data-sizes="(min-width:1024px) 23vw, (min-width:768px) 39vw, 83vw"          class="lazyload"
        
        alt="Using AI to predict wildfires: fire burning landscape"
        
                                      />

    
    
  
      </a>
  

</div>
          
          <div class="text-container">
                              
<div class="f--field f--eyebrow">

    
  <span>Science/Technology</span>


</div>
            
                            <h3>
                                      <a href="https://today.usc.edu/using-ai-to-predict-wildfires/" >
                  
                    USC scientists use AI to predict a wildfire’s next move

                                      </a>
                                  </h3>
                      </div>

        </div>
          </div>
  

  </div></div>


  
    






<div
  class="cc--component-container cc--spacer "

  
  
  
  
  
  
  >
  <div class="c--component c--spacer"
    
      >

    

  </div></div>



      


  
    






<div
  class="cc--component-container cc--three-column-info-with-ctas no-links"

  
  
  
  
  
  
  >
  <div class="c--component c--three-column-info-with-ctas"
    
      >

    
<div class="inner-wrapper">

          
<div class="f--field f--section-title">

    
  <h2>Hot Topics</h2>


</div>
  
      <div class="cards-container">
              <div class="card">

                      <h3>
                              <a
                  href="https://today.usc.edu/tag/research/"
                                  >
              
                Research That Saves Lives

                              </a>
                          </h3>
          
                          
<div class="f--field f--description">

    
  <p>As one of the nation’s leading research institutions, USC is a place where bold ideas and cross-disciplinary collaboration fuel discovery and innovation. Our researchers tackle society’s most urgent challenges, delivering solutions that save lives and improve communities around the world.</p>



</div>
          
          
          
        </div>
              <div class="card">

                      <h3>
                              <a
                  href="https://today.usc.edu/tag/alumni/"
                                  >
              
                More Than a Half Million Trojans

                              </a>
                          </h3>
          
                          
<div class="f--field f--description">

    
  <p>With the latest class of graduates, there are now more than a half million USC alumni worldwide. It’s a milestone reached by few institutions.</p>



</div>
          
          
          
        </div>
              <div class="card">

                      <h3>
                              <a
                  href="https://today.usc.edu/tag/artificial-intelligence/"
                                  >
              
                A Global Leader in AI

                              </a>
                          </h3>
          
                          
<div class="f--field f--description">

    
  <p>As the world looks more and more to artificial intelligence for help in countless ways, USC researchers are at the forefront of this groundbreaking field.</p>



</div>
          
          
          
        </div>
          </div>
  
  
</div>


  </div></div>

				</div>
			</div>
		</article>
	</div>
						</div>
					</div>
				</div>

								
			</div>
		</div>
		<!-- t--template t--content-page -->

	</section>
	<!-- tc--template-container tc--content-page -->

				</div>
			</div>
		</main>

					<footer id="footer" aria-label="Site Footer">
				          
        

    
    






<div
  class="cc--component-container cc--footer gold"

  
  
  
  
  
  
  >
  <div class="c--component c--footer"
    
      >

    
  <div class="footer-main">
    <div class="footer-branding">
      <div class="logo-locations">
        <a class="logo" href="/" aria-label="University of Southern California" rel="home">
          <svg enable-background="new 0 0 274 59" version="1.1" viewBox="0 0 274 59" xml:space="preserve" xmlns="http://www.w3.org/2000/svg"><style type="text/css">.st0{fill-rule:evenodd;clip-rule:evenodd;fill:#990000;}.st1{fill-rule:evenodd;clip-rule:evenodd;}.st2{filter:url(#b);}.st3{fill-rule:evenodd;clip-rule:evenodd;fill:#FFFFFF;}.st4{mask:url(#a);fill-rule:evenodd;clip-rule:evenodd;}</style><path class="st0" d="m54.2 7.3c0-1-0.2-1.5-1-1.5h-1.4c-0.2 0-0.2-0.1-0.2-0.3v-0.6c0-0.2 0.1-0.2 0.3-0.2 0.3 0 1.7 0.1 4.6 0.1 2.5 0 4-0.1 4.3-0.1 0.2 0 0.2 0.1 0.2 0.2v0.7c0 0.2-0.1 0.2-0.2 0.2h-1.3c-0.9 0-1.4 0.5-1.5 1.2 0 0.4-0.1 2.2-0.1 3.9v3.7c0 1.5 0 3.6 0.1 4.6 0.1 1.4 0.3 2.8 1 3.7 1 1.4 2.6 1.9 4.1 1.9 1.4 0 3.1-0.5 4-1.4 1.1-1.1 1.7-2.9 1.7-5.1v-5.4c0-2.4-0.2-5-0.3-5.7-0.1-0.8-0.6-1.3-1.2-1.3h-1.1c-0.2 0-0.2-0.1-0.2-0.2v-0.8c0-0.2 0.1-0.2 0.2-0.2 0.3 0 1.7 0.1 3.7 0.1 1.9 0 2.8-0.1 3.1-0.1s0.3 0.1 0.3 0.2v0.7c0 0.2 0 0.3-0.2 0.3h-1.1c-1.1 0-1.2 0.8-1.3 1.2s-0.1 4.5-0.1 4.9-0.1 5.6-0.2 6.8c-0.2 2.5-0.9 4.3-2.5 5.7-1.3 1.1-3.2 1.9-5.4 1.9-1.4 0-3.2-0.2-4.3-0.7-1.7-0.7-2.9-2.1-3.5-4-0.4-1.3-0.5-3.3-0.5-7.2v-7.2z"></path><path class="st0" d="m74.2 19.3c0.1 0 0.2 0.1 0.2 0.2 0.1 1 0.3 2.1 1.1 3.4 0.9 1.4 2.5 2.2 4 2.2 1.6 0 3.9-1 3.9-3.5 0-2.8-1.3-3.7-4.4-4.8-1-0.4-2.9-1.2-4.4-2.5-1-1-1.5-2.5-1.5-4.3 0-1.7 0.7-3.1 2.1-4.2 1.2-1 2.5-1.4 4.3-1.4 2 0 3.2 0.5 3.8 0.7 0.3 0.1 0.5 0.1 0.6-0.1l0.4-0.4c0.1-0.1 0.2-0.2 0.4-0.2s0.2 0.2 0.2 0.5c0 0.5 0.5 4.7 0.5 5 0 0.1-0.1 0.2-0.2 0.2l-0.5 0.2c-0.1 0-0.2 0-0.3-0.2-0.3-0.8-0.8-2-1.9-3.2-0.8-0.8-1.7-1.3-3.1-1.3-2.4 0-3.4 1.9-3.4 3 0 1 0.1 1.9 0.8 2.7 0.8 0.8 2.5 1.5 3.7 1.9 1.7 0.5 3.2 1.1 4.3 2.3 1 1 1.5 2.3 1.5 4.3 0 3.7-2.4 6.6-6.3 6.6-2.5 0-3.9-0.5-5.1-1.4-0.4-0.2-0.5-0.4-0.7-0.4s-0.3 0.2-0.5 0.5-0.3 0.4-0.5 0.4-0.4-0.2-0.4-0.5c0.1-0.4 0.5-5.2 0.5-5.6 0-0.2 0.1-0.2 0.2-0.2l0.7 0.1z"></path><path class="st0" d="m107.4 21.1c0.1 0.1 0.1 0.2 0.1 0.3-1.8 2.6-4.7 5-8.9 5-3.8 0-6-1.1-7.8-2.8-2.2-2-3.2-5.4-3.2-8.1 0-2.9 1-5.7 3.1-8 1.6-1.7 4.2-3.2 7.8-3.2 2.8 0 5 0.8 6 1.4 0.4 0.2 0.5 0.2 0.6 0l0.2-0.8c0.1-0.2 0.1-0.2 0.5-0.2s0.4 0.1 0.4 0.5c0 0.5 0.6 5.3 0.7 6.2 0 0.2 0 0.3-0.1 0.3l-0.6 0.2c-0.1 0.1-0.2-0.1-0.3-0.3-0.4-1.1-1.1-2.6-2.3-3.9-1.3-1.3-2.9-2.2-5.2-2.2-4.8 0-6.8 4.3-6.8 8.6 0 1.9 0.5 5.4 2.8 7.8 2 2.1 3.7 2.3 5.1 2.3 3.3 0 5.9-1.6 7.2-3.5 0.1-0.2 0.3-0.3 0.4-0.2l0.3 0.6z"></path><path class="st1" d="m116.7 25.8c-2-0.8-3.1-2.1-3.6-4-0.4-1.3-0.5-3.3-0.5-7.3v-7.2c0-1-0.2-1.5-1.1-1.5h-1.4c-0.2 0-0.2-0.1-0.2-0.3v-0.6c0-0.2 0-0.2 0.3-0.2s1.8 0.1 4.1 0.1 3.8-0.1 4.1-0.1c0.2 0 0.3 0.1 0.3 0.2v0.8c0 0.1 0 0.2-0.2 0.2h-1.5c-0.9 0-1.4 0.6-1.4 1.2 0 0.3-0.1 2.4-0.1 3.9v2c0 1.6 0.1 5.4 0.2 6.3 0 1.4 0.3 2.8 1 3.8 1.1 1.6 3 2.2 4.8 2.2 1.7 0 3.4-0.6 4.5-1.6 1.2-1.3 1.6-3.1 1.6-5.3v-6.4c0-2.3-0.1-4-0.1-4.8-0.1-0.8-0.6-1.3-1.3-1.3h-1.3c-0.2 0-0.2-0.1-0.2-0.2v-0.8c0-0.2 0-0.2 0.2-0.2 0.3 0 1.7 0.1 3.7 0.1 2.1 0 3-0.1 3.3-0.1s0.3 0.1 0.3 0.2v0.7c0 0.2 0 0.3-0.2 0.3h-1.2c-1.1 0-1.2 0.8-1.3 1.2-0.1 0.3-0.1 4.5-0.1 5 0 0.4-0.1 5.7-0.1 6.8-0.2 2.5-0.8 4.3-2.5 5.8-1.2 1.1-3 1.8-5.6 1.8-2-0.1-3.5-0.3-4.5-0.7"></path><path class="st1" d="m142.4 24.3c0 0.5 0.3 0.6 0.7 0.6h1c0.3 0 0.3 0.1 0.3 0.3v0.5c0 0.2-0.1 0.3-0.2 0.3-0.2 0-1.1-0.1-2.8-0.1-1.5 0-2.5 0.1-2.7 0.1s-0.2-0.1-0.2-0.3v-0.5c0-0.2 0.1-0.3 0.4-0.3h0.9c0.3 0 0.6-0.2 0.6-0.7 0-0.8 0.1-1.7 0.1-2.5v-4c0-1.2 0-2.1-0.9-2.7-0.7-0.5-1.4-0.5-2.2-0.5-0.5 0-1.5 0.2-2.4 1.2-0.6 0.7-0.7 1.5-0.7 2v6.3c0 0.8 0.1 1 0.7 1h0.7c0.2 0 0.3 0.1 0.3 0.3v0.5c0 0.2-0.1 0.3-0.2 0.3-0.2 0-1-0.1-2.6-0.1-1.5 0-2.5 0.1-2.7 0.1s-0.2-0.1-0.2-0.3v-0.5c0-0.2 0.1-0.3 0.3-0.3h0.9c0.4 0 0.6-0.1 0.6-0.6v-7c0-0.9-0.5-1.4-1.2-1.8l-0.2-0.1c-0.2-0.1-0.2-0.2-0.2-0.3v-0.2c0-0.1 0.1-0.2 0.3-0.3l2.6-1.4c0.2-0.1 0.4-0.2 0.5-0.2s0.2 0.1 0.2 0.4l0.2 1.8h0.1c0.9-1.1 2.6-2.2 4.1-2.2 0.9 0 1.8 0.2 2.5 0.7 0.9 0.6 1.6 2.1 1.6 2.8v7.7z"></path><path class="st1" d="m146.1 8.1c0-0.7 0.6-1.6 1.8-1.6 1 0 1.6 0.8 1.6 1.5 0 0.8-0.5 1.6-1.7 1.6-0.8 0.1-1.7-0.5-1.7-1.5zm1.3 9.1c0-1-0.2-1.6-1.7-1.8-0.2 0-0.3-0.1-0.3-0.2v-0.3c0-0.1 0.1-0.1 0.2-0.2 1-0.4 2.1-0.8 3.2-1.5 0.2-0.1 0.4-0.2 0.6-0.2s0.2 0.1 0.2 0.4c-0.1 0.5-0.1 2.4-0.1 4.9v3.3 2.6c0.1 0.6 0.1 0.8 0.5 0.8h1.4c0.2 0 0.3 0.1 0.3 0.3v0.5c0 0.2-0.1 0.3-0.3 0.3s-1.6-0.1-2.9-0.1c-1.7 0-2.9 0.1-3.2 0.1-0.2 0-0.2-0.1-0.2-0.2v-0.6c0-0.2 0.1-0.2 0.3-0.2h1.5c0.3 0 0.4-0.1 0.5-0.6v-2.7-4.6z"></path><path class="st1" d="m158.1 23.2c0.6-1.2 2.7-6.3 3-7.5 0.2-0.8 0-1.3-0.5-1.3h-0.6c-0.1 0-0.1-0.2-0.1-0.3v-0.5c0-0.2 0.1-0.2 0.2-0.2 0.4 0 1.1 0.1 2.4 0.1 1.2 0 1.9-0.1 2.2-0.1 0.2 0 0.3 0 0.3 0.3v0.5c0 0.2 0 0.3-0.1 0.3h-0.9c-0.4 0-0.7 0.1-0.9 0.3-0.4 0.4-1.9 3.7-2.4 4.8-0.4 0.9-2.6 5.9-2.7 6.1-0.2 0.4-0.4 0.7-0.7 0.7s-0.5-0.3-0.6-0.6c-0.1-0.2-2.2-5.8-2.3-6.1-0.1-0.2-1.3-3.7-1.5-4.1-0.4-0.9-0.7-1.1-1.2-1.1h-0.7c-0.1 0-0.2-0.1-0.2-0.3v-0.5c0-0.2 0.1-0.2 0.3-0.2 0.3 0 0.8 0.1 2.5 0.1 1.6 0 2.4-0.1 2.7-0.1 0.2 0 0.3 0.1 0.3 0.2v0.5c0 0.2 0 0.3-0.2 0.3h-0.6c-0.3 0-0.5 0.2-0.5 0.6 0 0.5 0.8 3 1 3.5 0.1 0.5 1.4 4 1.8 4.6"></path><path class="st1" d="m170.9 17.1c0.6 0 0.8-0.1 0.8-0.4 0-0.7-0.4-2.7-2.3-2.7-1.3 0-2.6 0.9-3.1 3.1h4.6zm-4.7 0.9c-0.1 0.3-0.1 0.7-0.1 1.4 0 2.9 1.9 5.1 4.2 5.1 1.8 0 3.1-1 3.6-2 0.1-0.1 0.2-0.2 0.3-0.1l0.4 0.2c0.1 0 0.1 0.1 0 0.4-0.5 1.2-2.1 3.3-4.8 3.3-1.7 0-3.6-0.5-4.7-2.1-1-1.5-1.3-2.6-1.3-4.7 0-1.2 0.3-3.2 1.9-4.8 1.3-1.3 2.8-1.6 4.3-1.6 3.4 0 4.4 3.2 4.4 4.4 0 0.4-0.4 0.5-0.8 0.5h-7.4z"></path><path class="st1" d="m177.2 18c0-1.5-0.2-1.8-1.1-2.3l-0.6-0.3c-0.2-0.1-0.2-0.1-0.2-0.2v-0.2c0-0.1 0.1-0.2 0.2-0.3l3-1.5c0.2-0.1 0.3-0.1 0.4-0.1 0.2 0 0.2 0.2 0.2 0.4l0.2 2.1h0.1c0.9-1.4 2.3-2.6 3.6-2.6 1 0 1.5 0.6 1.5 1.3 0 0.9-0.7 1.5-1.3 1.5-0.3 0-0.6-0.1-0.9-0.2-0.2-0.1-0.6-0.2-1-0.2-0.5 0-1.3 0.3-1.8 1.3-0.2 0.3-0.3 0.9-0.3 1v6.3c0 0.9 0.2 1.2 0.8 1.2h1.4c0.2 0 0.2 0.1 0.2 0.3v0.5c0 0.2 0 0.3-0.2 0.3s-1.6-0.1-3.2-0.1c-1.9 0-3.1 0.1-3.3 0.1-0.1 0-0.2-0.1-0.2-0.3v-0.5c0-0.2 0.1-0.3 0.3-0.3h1.2c0.5 0 0.7-0.2 0.7-0.6v-3.6-3z"></path><path class="st1" d="m185.3 25.4c-0.1-0.1-0.1-0.3-0.2-0.4l-0.5-2.8c0-0.1 0.1-0.2 0.2-0.2l0.4-0.1c0.1 0 0.2 0 0.3 0.2l0.7 1.5c0.6 1.1 1 1.9 2.6 1.9 1 0 2.4-0.6 2.4-2 0-1.3-0.7-2.1-2.9-2.8-1.8-0.5-3.5-1.4-3.5-3.8 0-2.1 1.9-3.7 4.2-3.7 1.4 0 2.2 0.4 2.8 0.6 0.4 0.2 0.5 0.4 0.5 0.5l0.4 2.1c0 0.1-0.1 0.2-0.2 0.3l-0.4 0.1c-0.1 0-0.2 0-0.3-0.1l-1-1.4c-0.4-0.6-0.9-1.2-2.1-1.2-1.1 0-2.3 0.7-2.3 2 0 1.7 1.4 2.1 2.9 2.5 2.2 0.6 3.6 1.5 3.6 3.8 0 2.4-2 3.9-4.5 3.9-1.6 0-2.8-0.6-3.1-0.9"></path><path class="st1" d="m194.5 8.1c0-0.7 0.6-1.6 1.8-1.6 1 0 1.6 0.8 1.6 1.5 0 0.8-0.5 1.6-1.7 1.6-0.7 0.1-1.7-0.5-1.7-1.5zm1.3 9.1c0-1-0.2-1.6-1.7-1.8-0.2 0-0.3-0.1-0.3-0.2v-0.3c0-0.1 0.1-0.1 0.2-0.2 1-0.4 2.1-0.8 3.2-1.5 0.2-0.1 0.4-0.2 0.6-0.2s0.2 0.1 0.2 0.4c-0.1 0.5-0.1 2.4-0.1 4.9v3.3 2.6c0.1 0.6 0.1 0.8 0.5 0.8h1.4c0.2 0 0.3 0.1 0.3 0.3v0.5c0 0.2-0.1 0.3-0.3 0.3s-1.6-0.1-2.9-0.1c-1.7 0-2.9 0.1-3.2 0.1-0.1 0-0.2-0.1-0.2-0.2v-0.6c0-0.2 0.1-0.2 0.3-0.2h1.5c0.3 0 0.4-0.1 0.5-0.6v-2.7-4.6z"></path><path class="st1" d="m204 22.2c0 2.2 1 2.6 1.6 2.6 0.7 0 1.6-0.1 2-0.4 0.2-0.1 0.3-0.1 0.4 0.1l0.2 0.3c0.1 0.1 0 0.2-0.1 0.3-0.3 0.3-1.5 1.2-3.3 1.2-1.5 0-2.9-1.4-2.9-2.5 0-1.2 0.1-3.9 0.1-4.3v-0.7c0-0.5 0-2.5 0.1-3.8h-1.8c-0.3 0-0.4-0.1-0.4-0.4 0-0.2 0.2-0.3 0.5-0.5 0.7-0.4 3-2.8 3.4-3.8 0.1-0.2 0.2-0.3 0.4-0.3s0.3 0.2 0.3 0.4l-0.2 3h4.2c0.2 0 0.3 0.1 0.2 0.3l-0.3 1c0 0.1-0.1 0.2-0.3 0.2h-3.9c-0.1 1.4-0.1 2.8-0.1 4.2v3.1z"></path><path class="st1" d="m209.1 32.5c0-0.6 0.7-1.2 1.4-1.2 0.6 0 1.1 0.3 1.4 0.3 0.2 0 0.4 0 0.6-0.2 0.4-0.5 2-4 2.4-5.1-0.1-0.7-3.3-10.1-3.6-10.7-0.4-0.7-0.6-1.3-1.4-1.3h-0.8c-0.2 0-0.2-0.1-0.2-0.3v-0.6c0-0.1 0-0.2 0.2-0.2 0.3 0 1 0.1 2.8 0.1 1.5 0 2.5-0.1 2.8-0.1 0.2 0 0.3 0.1 0.3 0.2v0.6c0 0.1 0 0.2-0.2 0.2h-0.6c-0.3 0-0.4 0.2-0.4 0.7 0.1 0.7 2.1 7.2 2.5 7.9h0.1c0.3-0.5 2.9-6.7 3.1-7.4s-0.1-1.1-0.5-1.1h-0.5c-0.2 0-0.2-0.1-0.2-0.2v-0.6c0-0.1 0-0.2 0.2-0.2 0.4 0 1 0.1 2.5 0.1 0.8 0 1.5-0.1 1.8-0.1 0.2 0 0.2 0.1 0.2 0.2v0.6c0 0.2 0 0.3-0.1 0.3h-0.5c-0.3 0-0.7 0.2-1 0.5-0.7 0.8-2.6 5.4-4.7 10.1-1.5 3.3-2.6 6.1-3.2 7.1-0.4 0.8-1 1.9-2.5 1.9-1.3 0.1-1.9-0.6-1.9-1.5"></path><path class="st1" d="m237.7 19.4c0-3.3-2.4-5.4-4-5.4-0.8 0-1.6 0.2-2.3 1-1.2 1.2-1.6 2.4-1.6 4.7 0 1.7 0.2 2.8 1.1 4.2 0.7 1 1.6 1.6 3 1.6 2.3-0.1 3.8-2.9 3.8-6.1m-8.7 4.8c-1.4-1.6-1.6-2.8-1.6-4.9 0-1.5 0.8-3.4 2.6-4.8 1.6-1.2 3-1.4 4-1.4 1.6 0 3.3 0.8 4.2 1.7 1.3 1.4 2 2.9 2 4.9 0 3.3-2.9 6.6-6.6 6.6-1.5 0-3.3-0.6-4.6-2.1"></path><path class="st1" d="m243.2 14.9h-2.1c-0.1 0-0.2-0.1-0.1-0.3l0.1-0.9c0-0.2 0.1-0.2 0.4-0.2h1.7v-0.6c0-3 0.8-4.9 2-6.2 1.6-1.7 3.2-2.4 4.9-2.4 1.4 0 2.7 0.5 2.7 1.7 0 0.7-0.8 1.3-1.3 1.3-0.3 0-0.6-0.1-0.8-0.2-0.3-0.2-0.6-0.5-0.7-0.7-0.4-0.5-0.8-0.7-1.4-0.7-0.8 0-1.6 0.2-2.2 1-1.1 1.4-1 4-1 6.1v0.6h3.6c0.2 0 0.2 0.2 0.2 0.3l-0.2 1c0 0.1-0.1 0.1-0.4 0.1h-3.3v2.7 6.6c0.1 0.5 0.1 0.8 0.8 0.8h1.7c0.3 0 0.3 0.1 0.3 0.2v0.5c0 0.2-0.1 0.4-0.2 0.4-0.2 0-1.3-0.1-3.6-0.1-1.6 0-3.2 0.1-3.4 0.1s-0.2-0.1-0.2-0.2v-0.7c0-0.2 0.1-0.2 0.3-0.2h1.4c0.6 0 0.7-0.4 0.7-1v-9z"></path><path class="st1" d="m56 52.9c-0.4-0.3-0.5-0.4-0.7-0.4s-0.3 0.2-0.5 0.5-0.3 0.4-0.5 0.4-0.3-0.2-0.3-0.5c0.1-0.7 0.4-4.5 0.5-5.4 0-0.2 0.1-0.2 0.2-0.2h0.4c0.1 0 0.2 0.1 0.2 0.2 0 1 0.1 1.9 1.1 3.5 0.9 1.4 2.5 2.2 4.2 2.2 1.8 0 4.1-1.1 4.1-3.8 0-3.1-2.2-4.2-4.6-5.1-0.9-0.3-3.1-1.1-4.4-2.4-1-1-1.5-2.4-1.5-4.4 0-1.5 0.9-3 1.9-3.8 1.1-1 2.6-1.4 4.2-1.4 1.7 0 3 0.5 3.5 0.7 0.3 0.2 0.4 0.1 0.6 0l0.4-0.4c0.2-0.1 0.2-0.2 0.4-0.2 0.1 0 0.2 0.2 0.2 0.5 0 1 0.4 4.1 0.4 4.8 0 0.1 0 0.2-0.1 0.2l-0.4 0.1c-0.1 0-0.1 0-0.2-0.1-0.2-0.9-0.6-2-1.9-3.3-0.8-0.9-1.9-1.3-3.3-1.3-2.7 0-3.6 2-3.6 3.5 0 0.8 0.2 1.8 0.9 2.6 0.9 0.9 2.5 1.5 3.4 1.8 2.1 0.7 3.7 1.2 4.8 2.5 0.8 0.9 1.3 2.1 1.3 4.1 0 3.5-2.2 6.6-6.1 6.6-2 0.1-3.5-0.6-4.6-1.5"></path><path class="st1" d="m78.4 47.5c0-3.3-2.4-5.4-4-5.4-0.8 0-1.6 0.2-2.3 1-1.2 1.2-1.6 2.4-1.6 4.7 0 1.7 0.2 2.8 1.1 4.2 0.7 1 1.6 1.6 3 1.6 2.3-0.1 3.8-2.9 3.8-6.1m-8.7 4.8c-1.4-1.6-1.6-2.7-1.6-4.9 0-1.5 0.8-3.4 2.6-4.8 1.6-1.2 3-1.4 4-1.4 1.6 0 3.3 0.8 4.2 1.7 1.3 1.4 2 2.9 2 4.9 0 3.3-2.9 6.6-6.6 6.6-1.5 0-3.3-0.6-4.6-2.1"></path><path class="st1" d="m93.3 47.9c0 1.4 0 2.9 0.1 3.5 0.1 0.9 0.4 1 0.8 1 0.2 0 0.6 0 1-0.1 0.2-0.1 0.3 0 0.4 0.1l0.1 0.4c0 0.2 0 0.2-0.1 0.3l-3.1 1c-0.2 0.1-0.4 0.1-0.6 0.1-0.1 0-0.2-0.1-0.3-0.4l-0.2-1.7h-0.1c-0.8 1.2-2.6 2.1-4.1 2.1-0.8 0-1.8-0.1-2.8-1.4-1.4-1.4-1.4-2.4-1.4-5.7v-3.5c0-0.8-0.4-1-1.1-1.1-0.5 0-0.7-0.1-0.7-0.2v-0.5c0-0.2 0.1-0.2 0.3-0.2h2.7c0.2 0 0.3 0 0.4-0.1h0.3c0.2 0 0.4 0.2 0.3 0.4 0 0.3-0.1 1.7-0.1 3.4v2.1c0 1.2-0.3 3.1 0.7 4.5 0.7 1.1 1.6 1.1 2.4 1.1s2-0.8 2.6-1.5c0.1-0.2 0.3-0.5 0.3-0.8v-7.1c-0.1-0.8-0.5-1-1.3-1-0.5 0-0.7-0.1-0.7-0.2v-0.5c0-0.2 0.1-0.3 0.4-0.3h2.8c0.2 0 0.4 0 0.5-0.1h0.4c0.2 0 0.3 0.2 0.3 0.3 0 0.3-0.2 1-0.2 4v2.1z"></path><path class="st1" d="m99.8 50.3c0 2.2 1 2.6 1.6 2.6 0.7 0 1.6-0.1 2-0.4 0.2-0.1 0.3-0.1 0.4 0.1l0.2 0.3c0.1 0.1 0 0.2-0.1 0.3-0.3 0.3-1.5 1.2-3.3 1.2-1.5 0-2.9-1.4-2.9-2.5 0-1.2 0.1-3.9 0.1-4.3v-0.7c0-0.5 0-2.5 0.1-3.8h-1.8c-0.3 0-0.4-0.1-0.4-0.4 0-0.2 0.2-0.3 0.5-0.5 0.7-0.4 3-2.8 3.4-3.8 0.1-0.2 0.2-0.3 0.4-0.3s0.3 0.2 0.3 0.4l-0.1 3h4.2c0.2 0 0.3 0.2 0.2 0.3l-0.3 1c0 0.1-0.1 0.2-0.3 0.2h-3.9c-0.1 1.4-0.1 2.8-0.1 4.2v3.1z"></path><path class="st1" d="m106.7 37.7c0-1.8-0.4-2.2-1.4-2.5l-0.5-0.2c-0.2-0.1-0.2-0.1-0.2-0.2v-0.3c0-0.1 0.1-0.1 0.4-0.3l3.1-1.6c0.2-0.1 0.5-0.2 0.7-0.2s0.3 0.2 0.2 0.7-0.1 6.6-0.1 8.1v1.9h0.1c1-1.3 2.8-1.9 4.1-1.9s2.1 0.4 3 1.2c1.3 1.3 1.3 2.8 1.3 4.4v1.7c0 1.7 0.1 3.5 0.1 4.1 0 0.4 0.5 0.4 1.1 0.4h0.5c0.2 0 0.3 0.1 0.3 0.3v0.4c0 0.2-0.1 0.4-0.2 0.4-0.2 0-1.2-0.1-2.9-0.1-1.5 0-2.5 0.1-2.8 0.1-0.2 0-0.3-0.1-0.3-0.3v-0.5c0-0.2 0-0.2 0.2-0.2h0.7c0.5 0 1.1-0.1 1.1-0.6v-3-2.4c0-1.7-0.1-2.9-1-3.7-0.6-0.6-1.2-0.8-2-0.8-0.9 0-1.9 0.3-2.8 1.3-0.3 0.3-0.6 0.9-0.6 1.5v6.5c0 1 0.1 1.2 0.7 1.2h0.9c0.2 0 0.3 0.1 0.3 0.2v0.5c0 0.2-0.1 0.3-0.2 0.3-0.2 0-1.5-0.1-2.8-0.1-1.5 0-2.6 0.1-2.9 0.1-0.1 0-0.2-0.1-0.2-0.3v-0.5c0-0.2 0.1-0.2 0.4-0.2h0.6c0.5 0 0.9-0.1 1-0.5 0.1-0.6 0.1-3.9 0.1-5.8v-9.1z"></path><path class="st1" d="m126.7 45.2c0.6 0 0.8-0.2 0.8-0.4 0-0.7-0.4-2.7-2.3-2.7-1.3 0-2.6 0.9-3.1 3.1h4.6zm-4.8 0.9c-0.1 0.3-0.1 0.7-0.1 1.4 0 2.9 1.9 5.1 4.3 5.1 1.8 0 3.1-1 3.6-2 0.1-0.1 0.2-0.2 0.3-0.1l0.4 0.2c0.1 0 0.1 0.1 0 0.4-0.5 1.2-2.1 3.3-4.8 3.3-1.7 0-3.6-0.5-4.7-2.1-1-1.5-1.3-2.6-1.3-4.7 0-1.2 0.3-3.2 1.9-4.8 1.3-1.3 2.8-1.6 4.3-1.6 3.4 0 4.4 3.2 4.4 4.4 0 0.4-0.4 0.5-0.8 0.5h-7.5z"></path><path class="st1" d="m133.2 46.1c0-1.5-0.2-1.8-1.1-2.3l-0.6-0.3c-0.2-0.1-0.2-0.1-0.2-0.2v-0.2c0-0.1 0.1-0.2 0.2-0.3l3-1.5c0.2-0.1 0.3-0.1 0.4-0.1 0.2 0 0.2 0.2 0.2 0.4l0.1 2.1h0.1c0.9-1.4 2.3-2.6 3.6-2.6 1 0 1.5 0.6 1.5 1.3 0 0.9-0.7 1.5-1.3 1.5-0.3 0-0.6-0.1-0.9-0.2-0.2-0.1-0.6-0.2-1-0.2-0.5 0-1.3 0.3-1.8 1.3-0.2 0.3-0.3 0.9-0.3 1v6.3c0 0.9 0.2 1.2 0.8 1.2h1.4c0.2 0 0.2 0.1 0.2 0.3v0.5c0 0.2 0 0.3-0.1 0.3-0.2 0-1.6-0.1-3.2-0.1-1.9 0-3.1 0.1-3.3 0.1-0.1 0-0.2-0.1-0.2-0.3v-0.5c0-0.2 0.1-0.3 0.3-0.3h1.2c0.4 0 0.7-0.2 0.7-0.6v-3.6-3z"></path><path class="st1" d="m153.3 52.4c0 0.5 0.3 0.6 0.7 0.6h1c0.3 0 0.3 0.1 0.3 0.3v0.5c0 0.2-0.1 0.3-0.2 0.3-0.2 0-1.1-0.1-2.8-0.1-1.5 0-2.5 0.1-2.7 0.1s-0.2-0.1-0.2-0.3v-0.5c0-0.2 0.1-0.3 0.4-0.3h0.9c0.3 0 0.6-0.1 0.6-0.7 0-0.8 0.1-1.7 0.1-2.5v-4c0-1.2 0-2.1-0.9-2.7-0.7-0.5-1.4-0.5-2.2-0.5-0.5 0-1.5 0.2-2.4 1.2-0.6 0.7-0.7 1.5-0.7 2v6.2c0 0.8 0.1 1 0.7 1h0.7c0.2 0 0.3 0.1 0.3 0.3v0.5c0 0.2-0.1 0.3-0.2 0.3-0.2 0-1-0.1-2.6-0.1-1.5 0-2.5 0.1-2.7 0.1-0.1 0-0.2-0.1-0.2-0.3v-0.5c0-0.2 0.1-0.3 0.3-0.3h0.9c0.4 0 0.6-0.1 0.6-0.6v-7c0-0.9-0.5-1.4-1.2-1.8l-0.2-0.1c-0.2-0.1-0.2-0.2-0.2-0.3v-0.1c0-0.2 0.1-0.2 0.3-0.3l2.6-1.4c0.2-0.1 0.4-0.2 0.5-0.2s0.2 0.1 0.2 0.4l0.2 1.8h0.1c0.9-1.1 2.6-2.2 4.1-2.2 0.9 0 1.8 0.2 2.5 0.7 0.9 0.6 1.6 2.1 1.6 2.8v7.7z"></path><path class="st1" d="m164.8 51.7c-2.2-2-3.2-5.5-3.2-8.4 0-2.8 0.9-5.7 3.1-7.9 1.5-1.6 4.2-3 7.5-3 2.7 0 5 1 6 1.6 0.4 0.2 0.5 0.2 0.6 0l0.2-0.9c0.1-0.2 0.1-0.2 0.4-0.2s0.3 0.1 0.3 0.5c0 0.6 0.6 5 0.7 5.8 0 0.2 0 0.3-0.1 0.3l-0.4 0.1c-0.1 0-0.2 0-0.3-0.2-0.4-1-1.1-2.4-2.3-3.7-1.3-1.3-2.8-2.3-5.2-2.3-5.4 0-7.5 4.8-7.5 9.1 0 1.7 0.5 5.1 2.9 7.7 2 2.3 3.7 2.5 5.7 2.5 3.4 0 6-1.9 7.4-3.8 0.1-0.2 0.1-0.2 0.2-0.1l0.4 0.3c0.1 0.1 0.1 0.1 0.1 0.3-1.8 2.6-4.7 5-8.9 5-3.6 0.1-5.8-1-7.6-2.7"></path><path class="st1" d="m188.7 47.4c-1.1 0.8-4 1.4-4 3.9 0 1.1 0.7 1.6 1.6 1.6s2.5-1 2.5-1.9v-3.6zm2.2 3.1c0 1.7 0.4 2.2 0.9 2.2s0.9-0.2 1-0.4 0.2-0.2 0.3-0.1l0.3 0.2c0.1 0.1 0.2 0.2 0 0.5-0.2 0.5-0.8 1.5-2.3 1.5-1.1 0-1.9-0.8-2.1-2h-0.1c-0.6 0.7-1.6 1.4-3 1.8-0.6 0.2-1.1 0.2-1.8 0.2s-1.8-0.6-1.8-2.5c0-1.4 0.6-2.6 3-3.6 1.3-0.6 2.9-1.3 3.4-1.7v-1.9c-0.1-0.5-0.1-1.3-0.4-1.7s-0.9-0.7-1.8-0.7c-0.6 0-1.5 0.2-1.9 0.6s-0.2 0.8-0.2 1.4c0 1-0.4 1.7-1.6 1.7-0.5 0-0.7-0.3-0.7-1 0-1 0.8-2.1 2.1-2.8s2.3-1 3.7-1 2 0.4 2.4 0.8c0.6 0.6 0.7 1.3 0.7 2.1v6.4z"></path><path class="st1" d="m198.5 45.1v7c0 0.5 0.1 0.9 0.8 0.9h1.3c0.3 0 0.3 0.1 0.3 0.3v0.5c0 0.2-0.1 0.3-0.2 0.3-0.2 0-1.5-0.1-3.4-0.1s-3.1 0.1-3.4 0.1c-0.1 0-0.2-0.1-0.2-0.3v-0.5c0-0.2 0.1-0.3 0.4-0.3h1.4c0.3 0 0.6-0.2 0.7-0.7v-7.8-1.9c0-2.1 0-4.8-0.1-5.6-0.1-1-0.5-1.5-1.5-1.8l-0.6-0.2c-0.1-0.1-0.2-0.1-0.2-0.1v-0.2c0-0.1 0.1-0.1 0.3-0.2 1-0.5 2.4-1 3.5-1.7 0.3-0.2 0.4-0.2 0.6-0.2s0.3 0.2 0.2 0.5c0 1.6-0.1 7.9-0.1 9.5v2.5z"></path><path class="st1" d="m202.6 36.2c0-0.7 0.6-1.6 1.8-1.6 1 0 1.6 0.8 1.6 1.5 0 0.8-0.5 1.6-1.7 1.6-0.8 0-1.7-0.6-1.7-1.5zm1.3 9.1c0-1-0.2-1.6-1.7-1.8-0.2 0-0.3-0.1-0.3-0.2v-0.3c0-0.1 0.1-0.1 0.2-0.2 1-0.4 2.1-0.8 3.2-1.5 0.2-0.1 0.4-0.2 0.6-0.2s0.2 0.1 0.2 0.4c-0.1 0.5-0.1 2.4-0.1 4.9v3.3 2.6c0.1 0.6 0.1 0.8 0.5 0.8h1.4c0.2 0 0.3 0.1 0.3 0.3v0.5c0 0.2-0.1 0.3-0.3 0.3s-1.6-0.1-2.9-0.1c-1.7 0-2.9 0.1-3.2 0.1-0.1 0-0.2-0.1-0.2-0.2v-0.6c0-0.2 0.1-0.2 0.3-0.2h1.5c0.3 0 0.4-0.1 0.5-0.6v-2.7-4.6z"></path><path class="st1" d="m211.5 43h-2.1c-0.1 0-0.2-0.1-0.1-0.3l0.1-0.9c0-0.2 0.1-0.2 0.4-0.2h1.7v-0.6c0-3 0.8-4.9 2-6.2 1.6-1.7 3.2-2.4 4.9-2.4 1.4 0 2.7 0.5 2.7 1.7 0 0.7-0.8 1.3-1.3 1.3-0.3 0-0.6-0.1-0.8-0.2-0.3-0.1-0.6-0.5-0.7-0.7-0.4-0.5-0.8-0.7-1.4-0.7-0.8 0-1.6 0.2-2.2 1-1.1 1.4-1 4-1 6.1v0.6h3.6c0.2 0 0.2 0.2 0.2 0.3l-0.1 1c0 0.1-0.2 0.1-0.4 0.1h-3.3v2.7 6.6c0.1 0.5 0.2 0.8 0.8 0.8h1.7c0.3 0 0.3 0.1 0.3 0.2v0.5c0 0.2-0.1 0.4-0.2 0.4-0.2 0-1.3-0.1-3.6-0.1-1.6 0-3.2 0.1-3.4 0.1s-0.2-0.1-0.2-0.2v-0.7c0-0.1 0.1-0.2 0.3-0.2h1.4c0.6 0 0.7-0.4 0.7-1v-9z"></path><path class="st1" d="m227.2 47.5c0-3.3-2.4-5.4-3.9-5.4-0.8 0-1.6 0.2-2.3 1-1.2 1.2-1.6 2.4-1.6 4.7 0 1.7 0.2 2.8 1.1 4.2 0.7 1 1.6 1.6 3 1.6 2.2-0.1 3.7-2.9 3.7-6.1m-8.6 4.8c-1.4-1.6-1.6-2.7-1.6-4.9 0-1.5 0.8-3.4 2.6-4.8 1.6-1.2 3-1.4 4-1.4 1.6 0 3.3 0.8 4.2 1.7 1.3 1.4 2 2.9 2 4.9 0 3.3-2.9 6.6-6.6 6.6-1.5 0-3.4-0.6-4.6-2.1"></path><path class="st1" d="m232.1 46.1c0-1.5-0.2-1.8-1.1-2.3l-0.6-0.3c-0.2-0.1-0.2-0.1-0.2-0.2v-0.2c0-0.1 0.1-0.2 0.2-0.3l3-1.5c0.2-0.1 0.3-0.1 0.4-0.1 0.2 0 0.2 0.2 0.2 0.4l0.2 2.1h0.1c0.9-1.4 2.3-2.6 3.6-2.6 1 0 1.5 0.6 1.5 1.3 0 0.9-0.7 1.5-1.3 1.5-0.3 0-0.6-0.1-0.9-0.2-0.2-0.1-0.6-0.2-1-0.2-0.5 0-1.3 0.3-1.8 1.3-0.2 0.3-0.3 0.9-0.3 1v6.3c0 0.9 0.2 1.2 0.8 1.2h1.4c0.2 0 0.2 0.1 0.2 0.3v0.5c0 0.2 0 0.3-0.1 0.3-0.2 0-1.6-0.1-3.2-0.1-1.9 0-3.1 0.1-3.3 0.1-0.1 0-0.2-0.1-0.2-0.3v-0.5c0-0.2 0.1-0.3 0.3-0.3h1.2c0.4 0 0.7-0.2 0.7-0.6v-3.6-3z"></path><path class="st1" d="m252.2 52.4c0 0.5 0.3 0.6 0.7 0.6h1c0.3 0 0.3 0.1 0.3 0.3v0.5c0 0.2-0.1 0.3-0.2 0.3-0.2 0-1.1-0.1-2.8-0.1-1.5 0-2.5 0.1-2.7 0.1s-0.2-0.1-0.2-0.3v-0.5c0-0.2 0.1-0.3 0.4-0.3h0.9c0.3 0 0.6-0.1 0.6-0.7 0-0.8 0.1-1.7 0.1-2.5v-4c0-1.2 0-2.1-0.9-2.7-0.7-0.5-1.4-0.5-2.2-0.5-0.5 0-1.5 0.2-2.4 1.2-0.6 0.7-0.7 1.5-0.7 2v6.2c0 0.8 0.1 1 0.7 1h0.7c0.2 0 0.3 0.1 0.3 0.3v0.5c0 0.2-0.1 0.3-0.2 0.3-0.2 0-1-0.1-2.6-0.1-1.5 0-2.5 0.1-2.7 0.1-0.1 0-0.2-0.1-0.2-0.3v-0.5c0-0.2 0.1-0.3 0.3-0.3h0.9c0.4 0 0.6-0.1 0.6-0.6v-7c0-0.9-0.5-1.4-1.2-1.8l-0.2-0.1c-0.2-0.1-0.2-0.2-0.2-0.3v-0.1c0-0.2 0.1-0.2 0.3-0.3l2.6-1.4c0.2-0.1 0.4-0.2 0.5-0.2s0.2 0.1 0.2 0.4l0.2 1.8h0.1c0.9-1.1 2.6-2.2 4.1-2.2 0.9 0 1.8 0.2 2.5 0.7 0.9 0.6 1.6 2.1 1.6 2.8v7.7z"></path><path class="st1" d="m255.8 36.2c0-0.7 0.6-1.6 1.8-1.6 1 0 1.6 0.8 1.6 1.5 0 0.8-0.5 1.6-1.7 1.6-0.7 0-1.7-0.6-1.7-1.5zm1.4 9.1c0-1-0.2-1.6-1.7-1.8-0.2 0-0.3-0.1-0.3-0.2v-0.3c0-0.1 0.1-0.1 0.2-0.2 1-0.4 2.1-0.8 3.2-1.5 0.2-0.1 0.4-0.2 0.6-0.2 0.1 0 0.2 0.1 0.1 0.4-0.1 0.5-0.1 2.4-0.1 4.9v3.3 2.6c0.1 0.6 0.1 0.8 0.5 0.8h1.4c0.2 0 0.3 0.1 0.3 0.3v0.5c0 0.2-0.1 0.3-0.3 0.3s-1.6-0.1-2.9-0.1c-1.7 0-2.9 0.1-3.2 0.1-0.1 0-0.2-0.1-0.2-0.2v-0.6c0-0.2 0.1-0.2 0.3-0.2h1.5c0.3 0 0.5-0.1 0.5-0.6v-2.7-4.6z"></path><path class="st1" d="m268.8 47.4c-1.1 0.8-4 1.4-4 3.9 0 1.1 0.7 1.6 1.6 1.6s2.5-1 2.5-1.9v-3.6zm2.2 3.1c0 1.7 0.4 2.2 0.9 2.2s0.9-0.2 1.1-0.4c0.1-0.2 0.2-0.2 0.3-0.1l0.3 0.2c0.1 0.1 0.2 0.2 0 0.5-0.2 0.5-0.8 1.5-2.3 1.5-1.1 0-1.9-0.8-2.1-2h-0.1c-0.6 0.7-1.6 1.4-3 1.8-0.6 0.2-1.1 0.2-1.8 0.2s-1.8-0.6-1.8-2.5c0-1.4 0.6-2.6 3-3.6 1.3-0.6 2.9-1.3 3.4-1.7v-1.9c-0.1-0.5-0.1-1.3-0.4-1.7s-0.9-0.7-1.8-0.7c-0.6 0-1.5 0.2-1.9 0.6s-0.2 0.8-0.2 1.4c0 1-0.4 1.7-1.6 1.7-0.5 0-0.7-0.3-0.7-1 0-1 0.8-2.1 2.1-2.8s2.3-1 3.7-1 2 0.4 2.4 0.8c0.6 0.6 0.7 1.3 0.7 2.1v6.4z"></path><defs><filter id="b" x="0" y="0" width="45.8" height="59.1" filterUnits="userSpaceOnUse"><feColorMatrix values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 1 0"></feColorMatrix></filter></defs><mask id="a" x="0" y="0" width="45.8" height="59.1" maskUnits="userSpaceOnUse"><g class="st2"><polygon class="st3" points="0 0 45.8 0 45.8 59.1 0 59.1"></polygon></g></mask><path class="st4" d="m45.8 12.9-0.1-1.4c-0.1-1.3-0.2-2.6-0.5-4l-0.2-0.9-0.3-1.5-0.1-0.4-0.4 0.1s-0.7 0.2-1.8 0.4l-2.7 0.3c-0.5 0-1.1 0.1-1.7 0.1-1.6 0-3.1-0.1-4.6-0.4l-0.8-0.2h-0.1s-0.8-0.2-1.9-0.6l-1.3-0.5c-1.3-0.5-2.7-1.2-4.2-2.2l-1.2-0.9c-0.2-0.2-0.5-0.4-0.7-0.6l-0.3-0.2-0.3 0.3s-0.2 0.2-0.7 0.5l-1.3 0.9c-1 0.7-2.5 1.5-4.3 2.2l-1.3 0.5c-0.6 0.2-1.2 0.4-1.9 0.6l-1.1 0.2c-1.2 0.3-2.5 0.4-4 0.4-0.7 0-1.4 0-2-0.1l-2.8-0.3c-1-0.2-1.7-0.3-1.7-0.3l-0.4-0.1-0.1 0.4c-0.1 0.4-0.2 0.8-0.2 1.2l-0.2 1.2c-0.2 1.3-0.4 2.6-0.4 3.9l-0.2 1.4v2 1.3c0.1 3.8 0.5 6.4 0.6 6.6 0 0.3 0.8 6.4 3.9 13.8 2.8 6.9 8.3 16.3 18.1 22.3l0.4 0.3 0.4-0.3c9.8-6 15.2-15.4 18-22.3 3-7.4 3.8-13.4 3.8-13.8 0-0.2 0.5-2.8 0.5-6.5v-1.3c0.1-0.8 0.1-1.4 0.1-2.1zm-5 8.4c-0.9 0-2.3 0.2-3.2 0.4l-0.7 0.3c-0.6 0.2-1.2 0.3-1.8 0.5l0.1 0.4-0.1-0.4c-1.4 0.3-2.4 0.5-3.1 0.5-1.3 0-3.3-0.2-5.6-1-0.6-0.2-1.4-0.5-2.2-0.6s-1.5-0.2-2.3-0.1c-0.9 0.1-1.8 0.4-2.5 0.7-2.3 0.8-4.3 1-5.6 1-1.1 0-2.7-0.4-4-0.8l-0.1 0.4 0.1-0.4-1.6-0.4c-0.8-0.3-2.2-0.4-3.2-0.4-1.3 0-2.5 0.3-3.7 0.9 0-0.2-0.1-0.5-0.1-0.9 1.2-0.6 2.5-1 3.8-1 0.9 0 2.2 0.2 2.9 0.4l1.6 0.5 0.1-0.4-0.1 0.4c1.5 0.4 3.1 0.8 4.3 0.8 1.4 0 3.5-0.2 5.9-1 0.7-0.2 1.5-0.5 2.4-0.6l0.8-0.1 1.2 0.1c0.7 0.1 1.5 0.4 2 0.6 2.4 0.8 4.5 1 5.9 1 0.8 0 1.9-0.2 3.4-0.6 0.6-0.2 1.3-0.4 1.9-0.5l0.6-0.2c0.7-0.2 2.1-0.4 3-0.4 1.4 0 2.7 0.3 3.9 1 0 0.4-0.1 0.6-0.1 0.8-1.3-0.6-2.6-0.9-3.9-0.9zm-40-5.1 13.5 1c-0.3 0.7-0.5 1.5-0.6 2.3-1.1 0-2.5-0.4-3.8-0.8l-1.6-0.5c-0.8-0.2-2.2-0.4-3.1-0.4-1.4 0-2.9 0.4-4.2 1-0.1-0.7-0.2-1.6-0.2-2.6zm22.2 1.7c-0.3 0-0.6 0-0.9 0.1-0.9 0.1-1.8 0.4-2.5 0.7-2 0.7-3.7 0.9-5 1 0.6-4.2 4.1-7.3 8.4-7.3 4.2 0 7.8 3.2 8.4 7.3-1.3-0.1-3-0.3-4.9-0.9-0.6-0.2-1.3-0.5-2.2-0.6-0.5-0.3-0.9-0.3-1.3-0.3zm17.8 1.7c-0.9 0-2.3 0.2-3.2 0.4l-0.6 0.2c-0.6 0.2-1.2 0.4-1.9 0.5l0.1 0.4-0.1-0.4c-0.9 0.2-2.2 0.6-3.2 0.6-1.3 0-3.3-0.2-5.6-1-0.6-0.2-1.3-0.4-2.2-0.6-0.5-0.1-0.9-0.1-1.3-0.1-0.3 0-0.6 0-0.9 0.1-0.9 0.1-1.8 0.4-2.5 0.7-2.3 0.8-4.3 1-5.6 1-1.1 0-2.7-0.4-4-0.8l-1.6-0.5c-0.8-0.3-2.2-0.5-3.2-0.5-1.4 0-2.7 0.3-3.9 1 0-0.3-0.1-0.6-0.1-0.9 1.2-0.6 2.7-1 4.1-1 0.9 0 2.2 0.2 2.9 0.4l1.6 0.5 0.1-0.4-0.1 0.4c1.5 0.4 3.1 0.8 4.3 0.8 1.4 0 3.5-0.2 5.9-1 0.7-0.2 1.5-0.5 2.4-0.6l0.8-0.1 1.2 0.1c0.7 0.1 1.5 0.4 2 0.6 2.4 0.8 4.5 1 5.8 1 0.8 0 1.9-0.2 3.4-0.6 0.6-0.2 1.3-0.4 1.9-0.5l0.6-0.2c0.7-0.2 2.1-0.4 3-0.4 1.4 0 2.8 0.4 4 1 0 0.3-0.1 0.6-0.1 0.9-1.3-0.7-2.6-1-4-1zm0.1-1.7c-0.9 0-2.3 0.2-3.2 0.4l-0.7 0.2c-0.6 0.2-1.2 0.4-1.8 0.5-0.8 0.2-2.1 0.5-3 0.5-0.1-0.8-0.3-1.6-0.6-2.3l13.5-1c0 0.9-0.1 1.8-0.1 2.6-1.3-0.6-2.8-0.9-4.1-0.9zm4.1-3-13.7 1.8-0.3-0.6 14-3.1v1.9zm-0.5-7c0.2 1.3 0.3 2.6 0.4 3.9l-14.2 4c-0.2-0.4-0.5-0.7-0.8-1.1l14.6-6.8zm-3.6-1.7c1.5-0.2 2.6-0.4 3.2-0.5 0.1 0.3 0.1 0.5 0.1 0.8l-14.6 7.8c-0.1-0.2-0.3-0.3-0.5-0.4l11.8-7.7zm-7.4-0.2c1.5 0.3 3 0.4 4.6 0.4h0.5l-9.8 7.2c-0.4-0.3-0.8-0.6-1.3-0.9l6-6.7zm-3.3-0.9c0.8 0.3 1.5 0.5 1.8 0.6l-5 6.7c-0.3-0.1-0.5-0.2-0.8-0.3l4-7zm-5.2-2.5c1.4 0.9 2.8 1.6 4 2.1l-3.3 7.2c-0.6-0.2-1.2-0.3-1.9-0.4l1.2-8.9zm-2.1-1.5c0.3 0.2 0.6 0.5 0.9 0.7l-0.5 9.7h-0.3-0.3l-0.7-9.7c0.4-0.3 0.7-0.6 0.9-0.7zm-2.1 1.5 1.2 8.9c-0.7 0.1-1.3 0.2-1.9 0.4l-3.3-7.2c1.6-0.7 3-1.5 4-2.1zm-5.2 2.5 4 6.9c-0.3 0.1-0.5 0.2-0.8 0.3l-5-6.7c0.6-0.1 1.2-0.3 1.8-0.5zm-7.6 1.3c1.5 0 2.9-0.1 4.3-0.4l6 6.7c-0.4 0.3-0.9 0.6-1.3 0.9l-9.9-7.2h0.9zm-6.3-0.7c0.6 0.1 1.7 0.3 3.1 0.5l11.8 7.6c-0.2 0.1-0.3 0.3-0.5 0.5l-14.5-7.8c0-0.3 0.1-0.5 0.1-0.8zm-0.4 2.2 14.5 6.7c-0.3 0.3-0.5 0.7-0.8 1l-14.2-4c0.2-1.1 0.3-2.4 0.5-3.7zm-0.5 5.1 14 3.1c-0.1 0.2-0.2 0.4-0.3 0.7l-13.7-1.9v-1.9zm39.9 23.2c-2.8 6.8-8.1 16.1-17.7 21.9-9.7-5.8-15-15.1-17.8-21.9-2.6-6.2-3.5-11.5-3.8-13.1 1.1-0.6 2.3-1 3.6-1 0.9 0 2.2 0.2 2.9 0.4l1.6 0.4c1.4 0.4 3 0.8 4.2 0.8 1.4 0 3.5-0.2 5.9-1 0.7-0.2 1.5-0.5 2.4-0.6l0.8-0.1 1.2 0.1c0.7 0.1 1.4 0.3 2 0.6 2.4 0.8 4.5 1 5.9 1 0.8 0 1.9-0.2 3.4-0.6 0.6-0.2 1.2-0.3 1.8-0.5l0.7-0.2c0.7-0.2 2.1-0.4 3-0.4 1.3 0 2.5 0.3 3.6 0.9-0.2 1.7-1.1 7-3.7 13.3z"></path><path class="st1" d="m34.6 30.5c0.5-0.3 1.1-1 0.8-2 0 0-0.7-1.3-0.4-2.4l0.1-0.4h-0.4c-0.1 0-0.6 0.1-1.1 0.7-0.6-0.9-1.5-1.5-1.6-1.5l-0.9-0.5 0.6 0.9c0.1 0.2 0.4 0.7 0.3 1-0.1 0.2-0.1 0.3-0.2 0.5-0.6-0.6-1.5-0.9-1.6-0.9l-0.5-0.1 0.2 0.5c0.3 0.6 0.1 1.1-0.2 1.5-0.2 0.4-0.4 0.8-0.4 1.3 0 0.7 0.3 1.1 0.6 1.3-0.2 0.2-0.4 0.5-0.4 0.8 0 0.4 0.2 0.8 0.6 1 0.1 0.6 0.3 0.9 0.6 1.1l0.3 7.4c-0.4 0.1-0.8 0.5-0.8 1s0.4 0.9 0.9 1l0.1 3.1c0 0.5 0.4 0.9 0.9 0.9s0.9-0.4 0.9-0.9l0.1-3.1c0.5-0.1 0.9-0.5 0.9-1s-0.4-0.9-0.8-1l0.3-7.4c0.1 0 0.2-0.1 0.3-0.2 0.2-0.2 0.3-0.5 0.4-1 0.1 0 0.2-0.1 0.2-0.2 0.2-0.2 0.3-0.5 0.3-0.8 0.3-0.1 0.1-0.4-0.1-0.6zm-0.6-3.5c0.2-0.3 0.4-0.4 0.6-0.5-0.1 1.1 0.5 2.2 0.5 2.2 0.3 1-0.7 1.4-0.9 1.5h-0.3-0.5c0.4-0.5 0.8-1.2 0.8-2.2-0.1-0.4-0.2-1-0.2-1zm-1.5-0.5c0.1-0.2 0.1-0.4 0-0.6 0.5 0.5 1.1 1.2 1.1 2 0 1.1-0.7 1.9-1 2.2h-0.2c-0.2-0.3-0.6-1-0.2-1.7 0.1-0.1 0.5-0.8 0.5-0.9 0.1-0.1 0.1-0.3 0.1-0.3l-0.2 0.2c-0.1 0.1-0.5 0.5-0.6 0.7-0.7 0.8-0.2 1.7 0 1.9h-0.6c-1-1.3 0.1-2.3 0.1-2.4 0.1-0.1 0.1-0.2 0.2-0.2 0.4-0.1 0.7-0.4 0.8-0.9zm-2.6 2.6c0-0.4 0.2-0.7 0.3-1.1 0.2-0.4 0.4-0.9 0.3-1.5 0.3 0.1 0.7 0.3 1 0.6-0.1 0.1-0.1 0.2-0.2 0.2 0 0-1.2 1.2-0.4 2.7h-0.2-0.3c0 0.2-0.5-0.1-0.5-0.9zm2.4 17.1c-0.2 0-0.4-0.2-0.4-0.4l-0.1-3h1l-0.1 3c0 0.2-0.2 0.4-0.4 0.4zm1.5-4.5c0 0.3-0.2 0.5-0.5 0.5h-1.8v0.2-0.2c-0.3 0-0.5-0.2-0.5-0.5s0.2-0.5 0.5-0.5h1 0.8c0.2 0 0.5 0.3 0.5 0.5zm-0.9-1h-1.2l-0.3-7.2h1.9l-0.4 7.2zm0.8-7.8c-0.1 0.1-0.3 0.2-0.5 0.2h-1.9c-0.3 0-0.5-0.2-0.6-0.6h3.1c0.1 0.1 0 0.3-0.1 0.4zm0.6-1.1c-0.1 0.1-0.3 0.2-0.5 0.2h-3.1c-0.4 0-0.6-0.3-0.7-0.6 0-0.4 0.3-0.7 0.6-0.7h0.5 0.9 1.9c0.4 0 0.6 0.3 0.6 0.6 0 0.2-0.1 0.3-0.2 0.5z"></path><path class="st1" d="m15.6 30.5c0.5-0.3 1.1-1 0.8-2 0 0-0.7-1.3-0.4-2.4l0.1-0.4h-0.4c-0.1 0-0.6 0.1-1.1 0.7-0.6-0.9-1.5-1.5-1.6-1.5l-0.9-0.5 0.6 0.9c0.1 0.2 0.4 0.7 0.3 1-0.1 0.2-0.1 0.3-0.2 0.5-0.6-0.6-1.5-0.9-1.6-0.9l-0.5-0.1 0.2 0.5c0.3 0.6 0.1 1.1-0.2 1.5-0.2 0.4-0.4 0.8-0.4 1.3 0 0.7 0.4 1.1 0.7 1.3-0.2 0.2-0.3 0.5-0.3 0.8 0 0.4 0.2 0.8 0.6 1 0 0.5 0.3 0.9 0.6 1.1l0.3 7.4c-0.4 0.1-0.8 0.5-0.8 1s0.4 0.9 0.9 1l0.1 3.1c0 0.5 0.4 0.9 0.9 0.9s0.9-0.4 0.9-0.9l0.1-3.1c0.2 0 0.4-0.1 0.6-0.3s0.3-0.4 0.3-0.7c0-0.5-0.4-0.9-0.8-1l0.3-7.4c0.1-0.1 0.2-0.1 0.3-0.2 0.2-0.2 0.3-0.5 0.4-0.9 0.3-0.2 0.6-0.6 0.6-1 0-0.2-0.2-0.5-0.4-0.7zm-0.6-3.5c0.2-0.3 0.4-0.4 0.6-0.5-0.1 1.1 0.5 2.2 0.5 2.2 0.3 1-0.7 1.4-0.9 1.5h-0.3-0.5c0.4-0.5 0.8-1.2 0.8-2.2-0.1-0.4-0.2-1-0.2-1zm-1.5-0.4c0.1-0.2 0.1-0.4 0-0.6 0.5 0.5 1.1 1.2 1.1 2 0 1.1-0.7 1.9-1 2.2h-0.2c-0.2-0.3-0.6-1-0.2-1.7 0.1-0.2 0.5-0.8 0.5-0.9 0.1-0.1 0.1-0.3 0.1-0.3l-0.2 0.2c-0.1 0.1-0.5 0.5-0.6 0.7-0.7 0.8-0.2 1.7 0 1.9h-0.6c-1-1.3 0.1-2.3 0.1-2.4 0.1-0.1 0.1-0.2 0.2-0.2 0.4-0.2 0.6-0.5 0.8-0.9zm-2.6 2.6c0-0.4 0.2-0.7 0.3-1.1 0.2-0.4 0.4-0.9 0.3-1.5 0.3 0.1 0.7 0.3 1 0.6-0.1 0.1-0.1 0.2-0.2 0.2 0 0-1.2 1.2-0.4 2.7h-0.2-0.2c0 0.1-0.6-0.1-0.6-0.9zm2.4 17c-0.2 0-0.4-0.2-0.4-0.4l-0.1-3h1.1l-0.1 3c-0.1 0.2-0.3 0.4-0.5 0.4zm1.4-4.4c0 0.1-0.1 0.3-0.2 0.4s-0.2 0.2-0.4 0.2h-1.8v0.2-0.2c-0.3 0-0.5-0.2-0.5-0.5s0.2-0.5 0.5-0.5h1 0.8c0.4-0.2 0.6 0.1 0.6 0.4zm-0.8-1h-1.2l-0.3-7.2h1.9l-0.4 7.2zm0.8-7.9c-0.1 0.1-0.3 0.2-0.5 0.2h-1.9c-0.3 0-0.5-0.2-0.6-0.6h3 0.1c0 0.2 0 0.3-0.1 0.4zm0.1-0.9h-3c-0.4 0-0.7-0.3-0.7-0.6 0-0.2 0.1-0.3 0.2-0.5 0.1-0.1 0.3-0.2 0.5-0.2h1.2 1 0.9c0.4 0 0.6 0.3 0.6 0.6 0 0.4-0.3 0.7-0.7 0.7z"></path><path class="st1" d="m25.9 31.3c0.5-0.4 1.2-1.2 0.8-2.6 0 0-0.6-1.4-0.1-2.9l0.1-0.4h-0.4s-0.7 0.1-1.4 0.9c-0.7-1.5-2.3-2.5-2.4-2.5l-0.8-0.5 0.5 0.8s0.5 0.9 0.3 1.5c-0.1 0.3-0.2 0.5-0.4 0.7-0.8-0.9-2.1-1.2-2.2-1.2l-0.4-0.1 0.2 0.4c0.4 0.9 0.1 1.5-0.3 2.2-0.3 0.5-0.5 1.1-0.5 1.8 0 1 0.5 1.6 1 1.9-0.4 0.3-0.6 0.7-0.6 1.2 0 0.6 0.3 1.1 0.8 1.3 0.1 0.6 0.4 1.1 0.9 1.4l0.5 10.6c-0.6 0.1-1.1 0.6-1.1 1.3s0.5 1.2 1.2 1.3l0.2 4.4c0 0.3 0.1 0.6 0.3 0.8s0.5 0.3 0.8 0.3 0.6-0.1 0.8-0.3 0.3-0.5 0.3-0.8l0.2-4.4c0.7 0 1.3-0.6 1.3-1.3s-0.5-1.2-1.1-1.3l0.5-10.6c0.5-0.3 0.8-0.8 0.9-1.4 0.1-0.1 0.3-0.2 0.4-0.3 0.3-0.3 0.4-0.7 0.4-1.1-0.1-0.4-0.3-0.9-0.7-1.1zm0.1-5.3c-0.3 1.5 0.2 2.7 0.2 2.8 0.4 1.4-0.6 2.1-0.9 2.2h-0.3-1c0.5-0.6 1.2-1.7 1.2-3.2 0-0.4-0.1-0.7-0.2-1 0.5-0.4 0.8-0.7 1-0.8zm-3.1-0.2c0.1-0.4 0.1-0.8 0-1.1 0.7 0.6 1.9 1.8 1.9 3.2 0 1.6-1 2.8-1.4 3.2h-0.1c-0.2-0.3-0.4-0.8-0.4-1.3 0-1.1 0.5-1.7 0.5-1.7s-0.7 0.4-0.8 1.4c-0.1 0.5 0 1 0.4 1.6h-0.6c-0.2-0.3-0.9-1.4-0.2-2.5 0.1-0.2 0.8-1.1 0.9-1.2 0.1-0.2 0.1-0.4 0.1-0.4-0.1 0.2-0.3 0.3-0.3 0.3-0.2 0.2-0.7 0.7-0.9 1-1 1.2-0.2 2.5 0 2.8h-0.4c-1.5-1.9 0.1-3.4 0.2-3.5l0.3-0.3c0.3-0.5 0.6-0.9 0.8-1.5zm-3.5 3.6c0-0.6 0.2-1.1 0.5-1.6 0.3-0.6 0.6-1.3 0.4-2.2 0.4 0.2 1.1 0.5 1.5 1l-0.5 0.5s-1.8 1.7-0.4 3.8h-0.2c-0.1 0-0.2 0-0.4 0.1 0 0-0.9-0.4-0.9-1.6zm4.1 23.4c0 0.2-0.1 0.3-0.2 0.4s-0.3 0.2-0.4 0.2c-0.2 0-0.3-0.1-0.4-0.2s-0.2-0.3-0.2-0.5l-0.2-4.3h1.7l-0.3 4.4zm1.5-5.7c0 0.5-0.4 0.8-0.8 0.8h-2.5c-0.5 0-0.8-0.4-0.8-0.8 0-0.5 0.4-0.8 0.8-0.8h1.9 0.6c0.4-0.1 0.8 0.3 0.8 0.8zm-1.1-1.3h-1.9l-0.5-10.4h0.1 2.6 0.2l-0.5 10.4zm0.3-10.9h-2.6c-0.4 0-0.8-0.5-0.9-0.9h0.1 4.3 0.2c-0.3 0.4-0.7 0.9-1.1 0.9zm1.6-1.7c-0.2 0.2-0.4 0.3-0.7 0.3h-4.3c-0.6 0-1-0.4-1-1s0.5-1 1-1h2.8 1.4c0.6 0 1 0.4 1 1 0 0.3-0.1 0.5-0.2 0.7z"></path></svg>
        </a>

                              <div class="location-info">
                                  
<div class="f--field f--link">

    
  

              <a class="link "  href=https://maps.usc.edu/?id=1928 target="_blank" aria-label="Read more about University Park Campus (Map)">University Park Campus (Map)</a>


</div>
              
                              <a class="phone" href="tel:213-740-2311">213-740-2311</a>
                          </div>
                      <div class="location-info">
                                  
<div class="f--field f--link">

    
  

              <a class="link "  href=https://maps.usc.edu/?id=1928#!ct/53722?s/  aria-label="Read more about Health Sciences Campus (Map)">Health Sciences Campus (Map)</a>


</div>
              
                              <a class="phone" href="tel:323-442-2000">323-442-2000</a>
                          </div>
                      <div class="location-info">
                                  
<div class="f--field f--link">

    
  

              <a class="link "  href=https://www.usc.edu/tommy-cam/  aria-label="Read more about Tommy Cam">Tommy Cam</a>


</div>
              
                          </div>
                      <div class="location-info">
                                  
<div class="f--field f--link">

    
  

              <a class="link "  href=https://www.usc.edu/hecuba-cam/  aria-label="Read more about Hecuba Cam">Hecuba Cam</a>


</div>
              
                          </div>
                        </div>

        
    






<div
  class="cc--component-container cc--social-links "

  
  
  
  
  
  
  >
  <div class="c--component c--social-links"
    
      >

    
      <ul class="social-media-links--platforms platforms" aria-label="Follow us on social media">
                        <li>
            <a href="https://www.facebook.com/usc/" class="ext" target="_blank" aria-label="Facebook (link is external)">
              <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
                width="32" height="32" viewBox="0 0 30 30" enable-background="new 0 0 30 30" xml:space="preserve">
                <path fill-rule="evenodd" clip-rule="evenodd" fill="#990000" d="M24,6H7C6.5,6,6,6.5,6,7v17c0,0.6,0.5,1,1,1
                  h9.1v-7.3h-2.5v-2.9h2.5v-2.1c0-2.5,1.5-3.8,3.7-3.8c0.7,0,1.5,0,2.2,0.1v2.6h-1.5c-1.2,0-1.4,0.6-1.4,1.4v1.8H22l-0.4,2.9h-2.5V25
                  H24c0.6,0,1-0.5,1-1V7C25,6.5,24.5,6,24,6z" />
              </svg>
            </a>
          </li>
        
        
        
        
        
        
        
                      
                  <li>
            <a href="https://twitter.com/usc" class="ext" target="_blank" aria-label="X (link is external)">
              <svg enable-background="new 0 0 30 30" viewBox="0 0 30 30" xmlns="http://www.w3.org/2000/svg">
                <path clip-rule="evenodd" d="m14.5 16.2-.8-1.1-6.3-9h2.7l5.1 7.3.8 1.1 6.6 9.4h-2.7zm2.4-2.8 7.4-8.7h-1.8l-6.5 7.5-5-7.4h-6l7.8 11.4-7.8 9h1.8l6.8-7.9 5.5 7.9h6z" fill-rule="evenodd" fill="#990000"/>
              </svg>
            </a>
          </li>
        
        
        
        
        
        
                      
        
                  <li>
            <a href="https://www.instagram.com/uscedu/" class="ext" target="_blank" aria-label="Instagram (link is external)">
              <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
              width="32" height="32" viewBox="0 0 30 30" enable-background="new 0 0 30 30" xml:space="preserve">
                <path fill-rule="evenodd" clip-rule="evenodd" fill="#990000" d="M15.5,6c-2.6,0-2.9,0-3.9,0.1
        c-1,0-1.7,0.2-2.3,0.4C8.7,6.7,8.1,7.1,7.6,7.6S6.7,8.7,6.5,9.3s-0.4,1.3-0.4,2.3C6,12.6,6,12.9,6,15.5s0,2.9,0.1,3.9
        c0,1,0.2,1.7,0.4,2.3c0.2,0.6,0.6,1.2,1.1,1.7s1.1,0.9,1.7,1.1s1.3,0.4,2.3,0.4s1.3,0.1,3.9,0.1s2.9,0,3.9-0.1c1,0,1.7-0.2,2.3-0.4
        c0.6-0.2,1.2-0.6,1.7-1.1s0.9-1.1,1.1-1.7s0.4-1.3,0.4-2.3s0.1-1.3,0.1-3.9s0-2.9-0.1-3.9c0-1-0.2-1.7-0.4-2.3
        c-0.2-0.6-0.6-1.2-1.1-1.7s-1.1-0.9-1.7-1.1s-1.3-0.4-2.3-0.4C18.4,6,18.1,6,15.5,6 M15.5,7.7c2.5,0,2.8,0,3.8,0.1
        c0.9,0,1.4,0.2,1.8,0.3c0.4,0.2,0.8,0.4,1.1,0.7c0.3,0.3,0.5,0.7,0.7,1.1c0.1,0.3,0.3,0.8,0.3,1.8s0.1,1.3,0.1,3.9
        c0,2.5,0,2.8-0.1,3.8c0,0.9-0.2,1.4-0.3,1.8c-0.2,0.4-0.4,0.8-0.7,1.1c-0.3,0.3-0.6,0.5-1.1,0.7c-0.3,0.1-0.8,0.3-1.8,0.3
        s-1.3,0.1-3.8,0.1s-2.8,0-3.8-0.1c-0.9,0-1.4-0.2-1.8-0.3c-0.4-0.2-0.8-0.4-1.1-0.7c-0.3-0.3-0.5-0.7-0.7-1.1
        c-0.1-0.3-0.3-0.8-0.3-1.8s-0.1-1.3-0.1-3.8s0-2.8,0.1-3.9c0-0.9,0.2-1.4,0.3-1.8c0.2-0.4,0.4-0.8,0.7-1.1s0.6-0.5,1.1-0.7
        c0.3-0.1,0.8-0.3,1.8-0.3C12.7,7.7,13,7.7,15.5,7.7 M15.5,18.7c-1.7,0-3.2-1.4-3.2-3.2s1.4-3.2,3.2-3.2s3.2,1.4,3.2,3.2
        S17.2,18.7,15.5,18.7 M15.5,10.6c-2.7,0-4.9,2.2-4.9,4.9s2.2,4.9,4.9,4.9s4.9-2.2,4.9-4.9S18.2,10.6,15.5,10.6 M21.7,10.4
        c0,0.6-0.5,1.1-1.1,1.1s-1.1-0.5-1.1-1.1s0.5-1.1,1.1-1.1S21.7,9.8,21.7,10.4" />
              </svg>
            </a>
          </li>
        
        
        
        
        
                      
        
        
        
                  <li>
            <a href="https://www.youtube.com/user/USC" class="ext" target="_blank" aria-label="Youtube (link is external)">
              <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 27 18.6" enable-background="new 0 0 27 18.6" xml:space="preserve">
              <path fill="#990000" d="M26.7,4c0,0-0.3-1.8-1.1-2.6c-1-1.1-2.2-1.1-2.7-1.1C19.2,0,13.5,0,13.5,0h0
                c0,0-5.7,0-9.4,0.3C3.5,0.3,2.4,0.3,1.3,1.4C0.5,2.2,0.3,4,0.3,4S0,6.1,0,8.3v2c0,2.1,0.3,4.3,0.3,4.3s0.3,1.8,1.1,2.6
                c1,1.1,2.4,1,3,1.1c2.2,0.2,9.2,0.3,9.2,0.3s5.7,0,9.5-0.3c0.5-0.1,1.7-0.1,2.7-1.1c0.8-0.8,1.1-2.6,1.1-2.6s0.3-2.1,0.3-4.3v-2
                C27,6.1,26.7,4,26.7,4z M10.7,13V5.5l7.2,3.6L10.7,13z"/>
              </svg>
            </a>
          </li>
        
        
        
                      
        
        
                  <li>
            <a href="https://www.linkedin.com/school/3084" class="ext" target="_blank" aria-label="LinkedIn (link is external)">
              <svg version="1.1" xmlns="http://www.w3.org/2000/svg"
                xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" width="32" height="32" viewBox="0 0 30 30"
                enable-background="new 0 0 30 30" xml:space="preserve">
                <path fill-rule="evenodd" clip-rule="evenodd" fill="#990000" d="M25,21.4c0,2-1.6,3.6-3.6,3.6H9.6
        c-2,0-3.6-1.6-3.6-3.6V9.6C6,7.6,7.6,6,9.6,6h11.8c2,0,3.6,1.6,3.6,3.6C25,9.6,25,21.4,25,21.4z M10.4,9.2c-1,0-1.6,0.6-1.6,1.5
        c-0.1,0.9,0.5,1.5,1.5,1.5l0,0c1,0,1.6-0.6,1.6-1.5S11.3,9.2,10.4,9.2z M11.8,21.9v-8.6H9v8.6H11.8z M22,21.9V17
        c0-2.6-1.4-3.8-3.3-3.8c-1.5,0-2.1,0.9-2.5,1.5l0,0v-1.2h-2.8c0,0,0,0.9,0,8.6h2.8v-4.9c0-0.2,0-0.5,0.1-0.7c0.1-0.5,0.6-1,1.5-1
        c1,0,1.5,0.7,1.5,2v4.6H22V21.9z" />
              </svg>
            </a>
          </li>
        
        
        
        
                  </ul>
  

  </div></div>
    </div>

    <div class="footer-links">
                
    






<div
  class="cc--component-container cc--footer-menu "

  
  
  
  
  
  
  >
  <div class="c--component c--footer-menu"
    
      >

    
      <div class="menu-container">
      <nav aria-label="Footer Navigation">
        <ul>
                      <li class="menu-item">

                              <span>USC News</span>
              
                              <div class="submenus-wrapper">
                                  <div class="submenus-wrapper-inner">
                    <ul class="submenu" aria-label="USC News Navigation">
                                              <li>
                          <a href="https://cloud.comms.usc.edu/preferences" >Subscribe to USC News</a>
                        </li>
                                          </ul>
                  </div>
                                </div>
                          </li>
                      <li class="menu-item">

                              <span>Trojan Family Magazine</span>
              
                              <div class="submenus-wrapper">
                                  <div class="submenus-wrapper-inner">
                    <ul class="submenu" aria-label="Trojan Family Magazine Navigation">
                                              <li>
                          <a href="https://today.usc.edu/trojan-family-magazine/class-notes-2/" >Class Notes</a>
                        </li>
                                              <li>
                          <a href="https://today.usc.edu/trojan-family-magazine/magazine-issues/" >Magazine Issues</a>
                        </li>
                                              <li>
                          <a href="https://today.usc.edu/trojan-family-magazine/connect/" >Connect with Trojan Family Magazine</a>
                        </li>
                                              <li>
                          <a href="https://cloud.comms.usc.edu/preferences" >Subscribe to Trojan Family Magazine</a>
                        </li>
                                              <li>
                          <a href="https://today.usc.edu/trojan-family-magazine/advertise/" >Advertise with Trojan Family Magazine</a>
                        </li>
                                          </ul>
                  </div>
                                </div>
                          </li>
                      <li class="menu-item">

                              <span>Pressroom</span>
              
                              <div class="submenus-wrapper">
                                  <div class="submenus-wrapper-inner">
                    <ul class="submenu" aria-label="Pressroom Navigation">
                                              <li>
                          <a href="https://today.usc.edu/pressroom/find-an-expert/" >Find an Expert</a>
                        </li>
                                              <li>
                          <a href="https://today.usc.edu/pressroom/media-contacts/" >Media Contacts</a>
                        </li>
                                              <li>
                          <a href="https://www.wrike.com/frontend/requestforms/index.html?token=eyJhY2NvdW50SWQiOjU2MDk5NzksInRhc2tGb3JtSWQiOjEzMTE5NDJ9CTQ5MjQ5NjE2NDg0NjEJZGVkYzRhNTBhOTA3ZjcxMmEyZTE3NTZmYTM5ZjFiOGNlMjQyMDI0YmQ2ZmU4OWQzYWJhYzhkZjI2MjYxMDY3Nw==" >Update Your Faculty Profile</a>
                        </li>
                                              <li>
                          <a href="https://today.usc.edu/pressroom/" >Pressroom</a>
                        </li>
                                          </ul>
                  </div>
                                </div>
                          </li>
                  </ul>
      </nav>
    </div>
  

  </div></div>
      
      <div class="footer-utility">
                    
    






<div
  class="cc--component-container cc--footer-menu-utility "

  
  
  
  
  
  
  >
  <div class="c--component c--footer-menu-utility"
    
      >

    
      <div class="menu-container">
      <nav aria-label="Privacy Navigation">
        <ul>
                      <li class="menu-item">

                              <a href="https://www.usc.edu/privacy-notice/" >Privacy Notice</a>
                          </li>
                      <li class="menu-item">

                              <a href="https://eeotix.usc.edu/notice-of-non-discrimination/" >Notice of Non-Discrimination</a>
                          </li>
                      <li class="menu-item">

                              <a href="https://accessibility.usc.edu/accessibility-at-usc/digital-accessibility/" >Digital Accessibility</a>
                          </li>
                  </ul>
      </nav>
    </div>
  

  </div></div>
        
                      
<div class="f--field f--description">

    
  <p>Copyright © 2026 University of Southern California</p>


</div>
              </div>
    </div>

    
  </div>


  </div></div>
			</footer>
			<script type="speculationrules">
{"prefetch":[{"source":"document","where":{"and":[{"href_matches":"/*"},{"not":{"href_matches":["/wp-*.php","/wp-admin/*","/wp-content/uploads/*","/wp-content/*","/wp-content/plugins/*","/wp-content/themes/usc-communications-2023/*","/*\\?(.+)"]}},{"not":{"selector_matches":"a[rel~=\"nofollow\"]"}},{"not":{"selector_matches":".no-prefetch, .no-prefetch a"}}]},"eagerness":"conservative"}]}
</script>
		<!-- Google Tag Manager (noscript) snippet added by Site Kit -->
		<noscript>
			<iframe src="https://www.googletagmanager.com/ns.html?id=GTM-NZS3SS3D" height="0" width="0" style="display:none;visibility:hidden"></iframe>
		</noscript>
		<!-- End Google Tag Manager (noscript) snippet added by Site Kit -->
		<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/vendor/jquery.min.js?ver=a292686bb14d2fb2e3065f" id="usc-script-jquery-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/vendor/jquery.once.min.js?ver=a292686bb14d2fb2e3065f" id="usc-script-jquery-once-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/vendor/what-input.min.js?ver=a292686bb14d2fb2e3065f" id="usc-script-what-input-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/vendor/ally.min.js?ver=a292686bb14d2fb2e3065f" id="usc-script-ally-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/vendor/lazysizes.min.js?ver=a292686bb14d2fb2e3065f" id="usc-script-lazysizes-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/vendor/jquery.hoverIntent.min.js?ver=a292686bb14d2fb2e3065f" id="usc-script-hover-intent-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/vendor/imagesloaded.pkgd.min.js?ver=a292686bb14d2fb2e3065f" id="usc-script-imagesloaded-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/vendor/gsap.min.js?ver=a292686bb14d2fb2e3065f" id="usc-gsap-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/vendor/ScrollTrigger.min.js?ver=a292686bb14d2fb2e3065f" id="usc-gsap-scroll-trigger-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/dist/DP.min.js?ver=a292686bb14d2fb2e3065f" id="usc-script-dp-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/dist/components/main-menu.min.js?ver=a292686bb14d2fb2e3065f" id="usc-script-main-menu-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/dist/components/header.min.js?ver=a292686bb14d2fb2e3065f" id="usc-script-header-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/dist/components/header-search.min.js?ver=a292686bb14d2fb2e3065f" id="usc-script-header-search-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/dist/components/stickyHeader.min.js?ver=a292686bb14d2fb2e3065f" id="usc-script-sticky-header-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/dist/components/landing-page-subnav.min.js?ver=a292686bb14d2fb2e3065f" id="usc-script-landing-page-subnav-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/js/notification-banner/notification-banner.js?ver=a292686bb14d2fb2e3065f" id="usc-notification-banner-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/js/page-alert-banner/page-alert-banner.js?ver=a292686bb14d2fb2e3065f" id="usc-page-alert-banner-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/dist/components/first-visit-notification.min.js?ver=a292686bb14d2fb2e3065f" id="usc-first-visit-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/dist/components/horizontal-strip.min.js?ver=a292686bb14d2fb2e3065f" id="usc-horizontal-strip-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/pattern_lab/source/js/dist/components/article-hero.min.js?ver=a292686bb14d2fb2e3065f" id="usc-article-hero-js"></script>
<script type="text/javascript" src="https://today.usc.edu/wp-content/themes/usc-communications-2023/js/countdown/countdown.js?ver=a292686bb14d2fb2e3065f" id="usc-countdown-js"></script>
<script id="wp-emoji-settings" type="application/json">
{"baseUrl":"https://s.w.org/images/core/emoji/17.0.2/72x72/","ext":".png","svgUrl":"https://s.w.org/images/core/emoji/17.0.2/svg/","svgExt":".svg","source":{"concatemoji":"https://today.usc.edu/wp-includes/js/wp-emoji-release.min.js?ver=6.9.4"}}
</script>
<script type="module">
/* <![CDATA[ */
/*! This file is auto-generated */
const a=JSON.parse(document.getElementById("wp-emoji-settings").textContent),o=(window._wpemojiSettings=a,"wpEmojiSettingsSupports"),s=["flag","emoji"];function i(e){try{var t={supportTests:e,timestamp:(new Date).valueOf()};sessionStorage.setItem(o,JSON.stringify(t))}catch(e){}}function c(e,t,n){e.clearRect(0,0,e.canvas.width,e.canvas.height),e.fillText(t,0,0);t=new Uint32Array(e.getImageData(0,0,e.canvas.width,e.canvas.height).data);e.clearRect(0,0,e.canvas.width,e.canvas.height),e.fillText(n,0,0);const a=new Uint32Array(e.getImageData(0,0,e.canvas.width,e.canvas.height).data);return t.every((e,t)=>e===a[t])}function p(e,t){e.clearRect(0,0,e.canvas.width,e.canvas.height),e.fillText(t,0,0);var n=e.getImageData(16,16,1,1);for(let e=0;e<n.data.length;e++)if(0!==n.data[e])return!1;return!0}function u(e,t,n,a){switch(t){case"flag":return n(e,"\ud83c\udff3\ufe0f\u200d\u26a7\ufe0f","\ud83c\udff3\ufe0f\u200b\u26a7\ufe0f")?!1:!n(e,"\ud83c\udde8\ud83c\uddf6","\ud83c\udde8\u200b\ud83c\uddf6")&&!n(e,"\ud83c\udff4\udb40\udc67\udb40\udc62\udb40\udc65\udb40\udc6e\udb40\udc67\udb40\udc7f","\ud83c\udff4\u200b\udb40\udc67\u200b\udb40\udc62\u200b\udb40\udc65\u200b\udb40\udc6e\u200b\udb40\udc67\u200b\udb40\udc7f");case"emoji":return!a(e,"\ud83e\u1fac8")}return!1}function f(e,t,n,a){let r;const o=(r="undefined"!=typeof WorkerGlobalScope&&self instanceof WorkerGlobalScope?new OffscreenCanvas(300,150):document.createElement("canvas")).getContext("2d",{willReadFrequently:!0}),s=(o.textBaseline="top",o.font="600 32px Arial",{});return e.forEach(e=>{s[e]=t(o,e,n,a)}),s}function r(e){var t=document.createElement("script");t.src=e,t.defer=!0,document.head.appendChild(t)}a.supports={everything:!0,everythingExceptFlag:!0},new Promise(t=>{let n=function(){try{var e=JSON.parse(sessionStorage.getItem(o));if("object"==typeof e&&"number"==typeof e.timestamp&&(new Date).valueOf()<e.timestamp+604800&&"object"==typeof e.supportTests)return e.supportTests}catch(e){}return null}();if(!n){if("undefined"!=typeof Worker&&"undefined"!=typeof OffscreenCanvas&&"undefined"!=typeof URL&&URL.createObjectURL&&"undefined"!=typeof Blob)try{var e="postMessage("+f.toString()+"("+[JSON.stringify(s),u.toString(),c.toString(),p.toString()].join(",")+"));",a=new Blob([e],{type:"text/javascript"});const r=new Worker(URL.createObjectURL(a),{name:"wpTestEmojiSupports"});return void(r.onmessage=e=>{i(n=e.data),r.terminate(),t(n)})}catch(e){}i(n=f(s,u,c,p))}t(n)}).then(e=>{for(const n in e)a.supports[n]=e[n],a.supports.everything=a.supports.everything&&a.supports[n],"flag"!==n&&(a.supports.everythingExceptFlag=a.supports.everythingExceptFlag&&a.supports[n]);var t;a.supports.everythingExceptFlag=a.supports.everythingExceptFlag&&!a.supports.flag,a.supports.everything||((t=a.source||{}).concatemoji?r(t.concatemoji):t.wpemoji&&t.twemoji&&(r(t.twemoji),r(t.wpemoji)))});
//# sourceURL=https://today.usc.edu/wp-includes/js/wp-emoji-loader.min.js
/* ]]> */
</script>

			<script type="text/javascript">window.NREUM||(NREUM={});NREUM.info={"beacon":"bam.nr-data.net","licenseKey":"NRJS-37496976d3a7f44e435","applicationID":"1128367822","transactionName":"M1VUbRBWVxVXUEJZCgofd1oWXlYIGUNXVwA=","queueTime":0,"applicationTime":831,"atts":"HxJXG1hMRBs=","errorBeacon":"bam.nr-data.net","agent":""}</script></body>
</html>

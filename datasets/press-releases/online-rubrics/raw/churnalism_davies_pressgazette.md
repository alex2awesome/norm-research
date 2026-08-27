<!-- URL: https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/ -->
<!-- HTTP: 200 SIZE:   167169 -->
<!DOCTYPE html>
<html lang="en-US">

<head>
    <meta charset="UTF-8" /><script type="text/javascript">(window.NREUM||(NREUM={})).init={privacy:{cookies_enabled:true},ajax:{deny_list:["bam.nr-data.net"]},feature_flags:["soft_nav"],distributed_tracing:{enabled:true}};(window.NREUM||(NREUM={})).loader_config={agentID:"601364689",accountID:"3767099",trustKey:"66686",xpid:"VwEBVlZaARAFU1BTDwcHVVw=",licenseKey:"NRJS-59243095c956cb51a60",applicationID:"571186038",browserID:"601364689"};;/*! For license information please see nr-loader-spa-1.314.0.min.js.LICENSE.txt */
(()=>{var e,t,r={384:(e,t,r)=>{"use strict";r.d(t,{NT:()=>a,Zm:()=>c,bQ:()=>u,dV:()=>d,pV:()=>l});var n=r(6154),i=r(1863),s=r(944),o=r(1910);const a={beacon:"bam.nr-data.net",errorBeacon:"bam.nr-data.net"};function c(){return n.gm.NREUM||(n.gm.NREUM={}),void 0===n.gm.newrelic&&(n.gm.newrelic=n.gm.NREUM),n.gm.NREUM}function d(){let e=c();return e.o||(e.o={ST:n.gm.setTimeout,SI:n.gm.setImmediate||n.gm.setInterval,CT:n.gm.clearTimeout,XHR:n.gm.XMLHttpRequest,REQ:n.gm.Request,EV:n.gm.Event,PR:n.gm.Promise,MO:n.gm.MutationObserver,FETCH:n.gm.fetch,WS:n.gm.WebSocket},(0,o.i)(...Object.values(e.o))),e}function u(e,t){let r=c();r.initializedAgents??={},t.initializedAt={ms:(0,i.t)(),date:new Date},r.initializedAgents[e]=t,2===Object.keys(r.initializedAgents).length&&(0,s.R)(69)}function l(){return function(){let e=c();const t=e.info||{};e.info={beacon:a.beacon,errorBeacon:a.errorBeacon,...t}}(),function(){let e=c();const t=e.init||{};e.init={...t}}(),d(),function(){let e=c();const t=e.loader_config||{};e.loader_config={...t}}(),c()}},782:(e,t,r)=>{"use strict";r.d(t,{T:()=>n});const n=r(860).K7.pageViewTiming},860:(e,t,r)=>{"use strict";r.d(t,{$J:()=>u,K7:()=>c,P3:()=>d,XX:()=>i,Yy:()=>a,df:()=>s,qY:()=>n,v4:()=>o});const n="events",i="jserrors",s="browser/blobs",o="rum",a="browser/logs",c={ajax:"ajax",genericEvents:"generic_events",jserrors:i,logging:"logging",metrics:"metrics",pageAction:"page_action",pageViewEvent:"page_view_event",pageViewTiming:"page_view_timing",sessionReplay:"session_replay",sessionTrace:"session_trace",softNav:"soft_navigations"},d={[c.pageViewEvent]:1,[c.pageViewTiming]:2,[c.metrics]:3,[c.jserrors]:4,[c.softNav]:5,[c.ajax]:6,[c.sessionTrace]:7,[c.sessionReplay]:8,[c.logging]:9,[c.genericEvents]:10},u={[c.pageViewEvent]:o,[c.pageViewTiming]:n,[c.ajax]:n,[c.softNav]:n,[c.metrics]:i,[c.jserrors]:i,[c.sessionTrace]:s,[c.sessionReplay]:s,[c.logging]:a,[c.genericEvents]:"ins"}},944:(e,t,r)=>{"use strict";r.d(t,{R:()=>i});var n=r(3241);function i(e,t){"function"==typeof console.debug&&(console.debug("New Relic Warning: https://github.com/newrelic/newrelic-browser-agent/blob/main/docs/warning-codes.md#".concat(e),t),(0,n.W)({drained:null,type:"data",name:"warn",feature:"warn",data:{code:e,secondary:t}}))}},993:(e,t,r)=>{"use strict";r.d(t,{A$:()=>s,ET:()=>o,TZ:()=>a,p_:()=>i});var n=r(860);const i={ERROR:"ERROR",WARN:"WARN",INFO:"INFO",DEBUG:"DEBUG",TRACE:"TRACE"},s={OFF:0,ERROR:1,WARN:2,INFO:3,DEBUG:4,TRACE:5},o="log",a=n.K7.logging},1687:(e,t,r)=>{"use strict";r.d(t,{Ak:()=>a,Ze:()=>d,x3:()=>c});var n=r(3241),i=r(3606),s=r(860),o=r(2646);function a(e,t){if(!e)return;const r={staged:!1,priority:s.P3[t]||0};e.runtime.drainRegistry.get(t)||e.runtime.drainRegistry.set(t,r)}function c(e,t){if(!e)return;const r=e.runtime.drainRegistry;r&&(r.get(t)&&r.delete(t),l(e,t,!1),r.size&&u(e))}function d(e,t="feature",r=!1){if(e){if(!e.runtime.drainRegistry.get(t)||r)return l(e,t);e.runtime.drainRegistry.get(t).staged=!0,u(e)}}function u(e){if(!e)return;const t=Array.from(e.runtime.drainRegistry);t.every(([e,t])=>t.staged)&&(t.sort((e,t)=>e[1].priority-t[1].priority),t.forEach(([t])=>{e.runtime.drainRegistry.delete(t),l(e,t)}))}function l(e,t,r=!0){if(!e)return;const s=e.ee,a=i.i.handlers;if(s&&!s.aborted&&s.backlog&&a){if((0,n.W)({type:"lifecycle",name:"drain",feature:t}),r){const e=s.backlog[t],r=a[t];if(r){for(let t=0;e&&t<e.length;++t)f(e[t],r);Object.entries(r).forEach(([e,t])=>{Object.values(t||{}).forEach(t=>{t[0]?.on&&t[0].context()instanceof o.y&&!t[0].listeners(e).includes(t[1])&&t[0].on(e,t[1])})})}}s.isolatedBacklog||delete a[t],s.backlog[t]=null,s.emit("drain-"+t,[])}}function f(e,t){var r=e[1];Object.values(t[r]||{}).forEach(t=>{var r=e[0];if(t[0]===r){var n=t[1],i=e[3],s=e[2];n.apply(i,s)}})}},1738:(e,t,r)=>{"use strict";r.d(t,{U:()=>f,Y:()=>l});var n=r(3241),i=r(9908),s=r(1863),o=r(944),a=r(3969),c=r(8362),d=r(860),u=r(4261);function l(e,t,r,s){const l=s||r;!l||l[e]&&l[e]!==c.d.prototype[e]||(l[e]=function(){(0,i.p)(a.xV,["API/"+e+"/called"],void 0,d.K7.metrics,r.ee),(0,n.W)({drained:!!r.runtime?.activatedFeatures,type:"data",name:"api",feature:u.Pl+e,data:{}});try{return t.apply(this,arguments)}catch(e){(0,o.R)(23,e)}})}function f(e,t,r,n,o){const a=e.info;null===r?delete a.jsAttributes[t]:a.jsAttributes[t]=r,(o||null===r)&&(0,i.p)(u.Pl+n,[(0,s.t)(),t,r],void 0,"session",e.ee)}},1741:(e,t,r)=>{"use strict";r.d(t,{W:()=>s});var n=r(944),i=r(4261);class s{#e(e,...t){if(this[e]!==s.prototype[e])return this[e](...t);(0,n.R)(35,e)}addPageAction(e,t){return this.#e(i.hG,e,t)}register(e){return this.#e(i.eY,e)}recordCustomEvent(e,t){return this.#e(i.fF,e,t)}setPageViewName(e,t){return this.#e(i.Fw,e,t)}setCustomAttribute(e,t,r){return this.#e(i.cD,e,t,r)}noticeError(e,t){return this.#e(i.o5,e,t)}setUserId(e,t=!1){return this.#e(i.Dl,e,t)}setApplicationVersion(e){return this.#e(i.nb,e)}setErrorHandler(e){return this.#e(i.bt,e)}addRelease(e,t){return this.#e(i.k6,e,t)}log(e,t){return this.#e(i.$9,e,t)}start(){return this.#e(i.d3)}finished(e){return this.#e(i.BL,e)}recordReplay(){return this.#e(i.CH)}pauseReplay(){return this.#e(i.Tb)}addToTrace(e){return this.#e(i.U2,e)}setCurrentRouteName(e){return this.#e(i.PA,e)}interaction(e){return this.#e(i.dT,e)}wrapLogger(e,t,r){return this.#e(i.Wb,e,t,r)}measure(e,t){return this.#e(i.V1,e,t)}consent(e){return this.#e(i.Pv,e)}}},1863:(e,t,r)=>{"use strict";function n(){return Math.floor(performance.now())}r.d(t,{t:()=>n})},1910:(e,t,r)=>{"use strict";r.d(t,{i:()=>s});var n=r(944);const i=new Map;function s(...e){return e.every(e=>{if(i.has(e))return i.get(e);const t="function"==typeof e?e.toString():"",r=t.includes("[native code]"),s=t.includes("nrWrapper");return r||s||(0,n.R)(64,e?.name||t),i.set(e,r),r})}},2555:(e,t,r)=>{"use strict";r.d(t,{D:()=>a,f:()=>o});var n=r(384),i=r(8122);const s={beacon:n.NT.beacon,errorBeacon:n.NT.errorBeacon,licenseKey:void 0,applicationID:void 0,sa:void 0,queueTime:void 0,applicationTime:void 0,ttGuid:void 0,user:void 0,account:void 0,product:void 0,extra:void 0,jsAttributes:{},userAttributes:void 0,atts:void 0,transactionName:void 0,tNamePlain:void 0};function o(e){try{return!!e.licenseKey&&!!e.errorBeacon&&!!e.applicationID}catch(e){return!1}}const a=e=>(0,i.a)(e,s)},2614:(e,t,r)=>{"use strict";r.d(t,{BB:()=>o,H3:()=>n,g:()=>d,iL:()=>c,tS:()=>a,uh:()=>i,wk:()=>s});const n="NRBA",i="SESSION",s=144e5,o=18e5,a={STARTED:"session-started",PAUSE:"session-pause",RESET:"session-reset",RESUME:"session-resume",UPDATE:"session-update"},c={SAME_TAB:"same-tab",CROSS_TAB:"cross-tab"},d={OFF:0,FULL:1,ERROR:2}},2646:(e,t,r)=>{"use strict";r.d(t,{y:()=>n});class n{constructor(e){this.contextId=e}}},2843:(e,t,r)=>{"use strict";r.d(t,{G:()=>s,u:()=>i});var n=r(3878);function i(e,t=!1,r,i){(0,n.DD)("visibilitychange",function(){if(t)return void("hidden"===document.visibilityState&&e());e(document.visibilityState)},r,i)}function s(e,t,r){(0,n.sp)("pagehide",e,t,r)}},3241:(e,t,r)=>{"use strict";r.d(t,{W:()=>s});var n=r(6154);const i="newrelic";function s(e={}){try{n.gm.dispatchEvent(new CustomEvent(i,{detail:e}))}catch(e){}}},3304:(e,t,r)=>{"use strict";r.d(t,{A:()=>s});var n=r(7836);const i=()=>{const e=new WeakSet;return(t,r)=>{if("object"==typeof r&&null!==r){if(e.has(r))return;e.add(r)}return r}};function s(e){try{return JSON.stringify(e,i())??""}catch(e){try{n.ee.emit("internal-error",[e])}catch(e){}return""}}},3333:(e,t,r)=>{"use strict";r.d(t,{$v:()=>u,TZ:()=>n,Xh:()=>c,Zp:()=>i,kd:()=>d,mq:()=>a,nf:()=>o,qN:()=>s});const n=r(860).K7.genericEvents,i=["auxclick","click","copy","keydown","paste","scrollend"],s=["focus","blur"],o=4,a=1e3,c=2e3,d=["PageAction","UserAction","BrowserPerformance"],u={RESOURCES:"experimental.resources",REGISTER:"register"}},3434:(e,t,r)=>{"use strict";r.d(t,{Jt:()=>o,YM:()=>u});var n=r(7836),i=r(5607),s=r(5732);const o="nr@original:".concat(i.W),a=50;var c=Object.prototype.hasOwnProperty,d=!1;function u(e,t,r){return e||(e=n.ee),i.inPlace=function(e,t,r,n,s,o){r||(r="");const a="-"===r.charAt(0);for(let c=0;c<t.length;c++){const d=t[c],u=e[d];f(u)||(e[d]=i(u,a?d+r:r,n,d,s,o))}},i.flag=o,i;function i(t,n,i,d,h,p){return f(t)?t:(n||(n=""),nrWrapper[o]=t,function(e,t,r){if(Object.defineProperty&&Object.keys)try{return Object.keys(e).forEach(function(r){Object.defineProperty(t,r,{get:function(){return e[r]},set:function(t){return e[r]=t,t}})}),t}catch(e){l([e],r)}for(var n in e)c.call(e,n)&&(t[n]=e[n])}(t,nrWrapper,e),nrWrapper);function nrWrapper(){var o,c,f,g;let m,v;try{c=this,o=[...arguments],v=p?(0,s.$5)(r):[void 0],f="function"==typeof i?i(o,c):i||{}}catch(t){l([t,"",[o,c,d],f],e)}u(n+"start",[o,c,d,v],f,h);const y=performance.now();let b;try{return g=t.apply(c,o),b=performance.now(),g}catch(e){throw b=performance.now(),u(n+"err",[o,c,e,v],f,h),m=e,m}finally{const e=b-y,t={start:y,end:b,duration:e,isLongTask:e>=a,methodName:d,thrownError:m};t.isLongTask&&u("long-task",[t,c,v],f,h),u(n+"end",[o,c,g,v],f,h)}}}function u(r,n,i,s){if(!d||t){var o=d;d=!0;try{e.emit(r,n,i,t,s)}catch(t){l([t,r,n,i],e)}d=o}}}function l(e,t){t||(t=n.ee);try{t.emit("internal-error",e)}catch(e){}}function f(e){return!(e&&"function"==typeof e&&e.apply&&!e[o])}},3606:(e,t,r)=>{"use strict";r.d(t,{i:()=>s});var n=r(9908);s.on=o;var i=s.handlers={};function s(e,t,r,s){o(s||n.d,i,e,t,r)}function o(e,t,r,i,s){s||(s="feature"),e||(e=n.d);var o=t[s]=t[s]||{};(o[r]=o[r]||[]).push([e,i])}},3738:(e,t,r)=>{"use strict";r.d(t,{He:()=>i,Kp:()=>a,Lc:()=>d,Rz:()=>u,TZ:()=>n,bD:()=>s,d3:()=>o,jx:()=>l,sl:()=>f,uP:()=>c});const n=r(860).K7.sessionTrace,i="bstResource",s="resource",o="-start",a="-end",c="fn"+o,d="fn"+a,u="pushState",l=1e3,f=3e4},3785:(e,t,r)=>{"use strict";r.d(t,{R:()=>c,b:()=>d});var n=r(9908),i=r(1863),s=r(860),o=r(3969),a=r(993);function c(e,t,r={},c=a.p_.INFO,d=!0,u,l=(0,i.t)()){(0,n.p)(o.xV,["API/logging/".concat(c.toLowerCase(),"/called")],void 0,s.K7.metrics,e),(0,n.p)(a.ET,[l,t,r,c,d,u],void 0,s.K7.logging,e)}function d(e){return"string"==typeof e&&Object.values(a.p_).some(t=>t===e.toUpperCase().trim())}},3878:(e,t,r)=>{"use strict";function n(e,t){return{capture:e,passive:!1,signal:t}}function i(e,t,r=!1,i){window.addEventListener(e,t,n(r,i))}function s(e,t,r=!1,i){document.addEventListener(e,t,n(r,i))}r.d(t,{DD:()=>s,jT:()=>n,sp:()=>i})},3962:(e,t,r)=>{"use strict";r.d(t,{AM:()=>o,O2:()=>l,OV:()=>s,Qu:()=>f,TZ:()=>c,ih:()=>h,pP:()=>a,t1:()=>u,tC:()=>i,wD:()=>d});var n=r(860);const i=["click","keydown","submit"],s="popstate",o="api",a="initialPageLoad",c=n.K7.softNav,d=5e3,u=500,l={INITIAL_PAGE_LOAD:"",ROUTE_CHANGE:1,UNSPECIFIED:2},f={INTERACTION:1,AJAX:2,CUSTOM_END:3,CUSTOM_TRACER:4},h={IP:"in progress",PF:"pending finish",FIN:"finished",CAN:"cancelled"}},3969:(e,t,r)=>{"use strict";r.d(t,{TZ:()=>n,XG:()=>a,rs:()=>i,xV:()=>o,z_:()=>s});const n=r(860).K7.metrics,i="sm",s="cm",o="storeSupportabilityMetrics",a="storeEventMetrics"},4234:(e,t,r)=>{"use strict";r.d(t,{W:()=>i});var n=r(1687);class i{constructor(e,t){this.agentRef=e,this.ee=e?.ee,this.featureName=t,this.blocked=!1}deregisterDrain(){(0,n.x3)(this.agentRef,this.featureName)}}},4261:(e,t,r)=>{"use strict";r.d(t,{$9:()=>u,BL:()=>c,CH:()=>p,Dl:()=>R,Fw:()=>w,PA:()=>v,Pl:()=>n,Pv:()=>x,Tb:()=>f,U2:()=>o,V1:()=>A,Wb:()=>T,bt:()=>b,cD:()=>y,d3:()=>E,dT:()=>d,eY:()=>g,fF:()=>h,hG:()=>s,hw:()=>i,k6:()=>a,nb:()=>m,o5:()=>l});const n="api-",i=n+"ixn-",s="addPageAction",o="addToTrace",a="addRelease",c="finished",d="interaction",u="log",l="noticeError",f="pauseReplay",h="recordCustomEvent",p="recordReplay",g="register",m="setApplicationVersion",v="setCurrentRouteName",y="setCustomAttribute",b="setErrorHandler",w="setPageViewName",R="setUserId",E="start",T="wrapLogger",A="measure",x="consent"},5205:(e,t,r)=>{"use strict";r.d(t,{j:()=>x});var n=r(384),i=r(1741);var s=r(2555),o=r(3333);const a=e=>{if(!e||"string"!=typeof e)return!1;try{document.createDocumentFragment().querySelector(e)}catch{return!1}return!0};var c=r(2614),d=r(944),u=r(8122);const l="[data-nr-mask]",f=e=>(0,u.a)(e,(()=>{const e={feature_flags:[],experimental:{register:!1,resources:!1},mask_selector:"*",block_selector:"[data-nr-block]",mask_input_options:{color:!1,date:!1,"datetime-local":!1,email:!1,month:!1,number:!1,range:!1,search:!1,tel:!1,text:!1,time:!1,url:!1,week:!1,textarea:!1,select:!1,password:!0}};return{ajax:{deny_list:void 0,block_internal:!0,enabled:!0,autoStart:!0},api:{register:{get enabled(){return e.feature_flags.includes(o.$v.REGISTER)||e.experimental.register},set enabled(t){e.experimental.register=t},duplicate_data_to_container:!1}},browser_consent_mode:{enabled:!1},distributed_tracing:{enabled:void 0,exclude_newrelic_header:void 0,cors_use_newrelic_header:void 0,cors_use_tracecontext_headers:void 0,allowed_origins:void 0},get feature_flags(){return e.feature_flags},set feature_flags(t){e.feature_flags=t},generic_events:{enabled:!0,autoStart:!0},harvest:{interval:30},jserrors:{enabled:!0,autoStart:!0},logging:{enabled:!0,autoStart:!0},metrics:{enabled:!0,autoStart:!0},obfuscate:void 0,page_action:{enabled:!0},page_view_event:{enabled:!0,autoStart:!0},page_view_timing:{enabled:!0,autoStart:!0},performance:{capture_marks:!1,capture_measures:!1,capture_detail:!0,resources:{get enabled(){return e.feature_flags.includes(o.$v.RESOURCES)||e.experimental.resources},set enabled(t){e.experimental.resources=t},asset_types:[],first_party_domains:[],ignore_newrelic:!0}},privacy:{cookies_enabled:!0},proxy:{assets:void 0,beacon:void 0},session:{expiresMs:c.wk,inactiveMs:c.BB},session_replay:{autoStart:!0,enabled:!1,preload:!1,sampling_rate:10,error_sampling_rate:100,collect_fonts:!1,inline_images:!1,fix_stylesheets:!0,mask_all_inputs:!0,get mask_text_selector(){return e.mask_selector},set mask_text_selector(t){a(t)?e.mask_selector="".concat(t,",").concat(l):""===t||null===t?e.mask_selector=l:(0,d.R)(5,t)},get block_class(){return"nr-block"},get ignore_class(){return"nr-ignore"},get mask_text_class(){return"nr-mask"},get block_selector(){return e.block_selector},set block_selector(t){a(t)?e.block_selector+=",".concat(t):""!==t&&(0,d.R)(6,t)},get mask_input_options(){return e.mask_input_options},set mask_input_options(t){t&&"object"==typeof t?e.mask_input_options={...t,password:!0}:(0,d.R)(7,t)}},session_trace:{enabled:!0,autoStart:!0},soft_navigations:{enabled:!0,autoStart:!0},ssl:void 0,user_actions:{enabled:!0,elementAttributes:["id","className","tagName","type"]}}})());var h=r(6154),p=r(9324);let g=0;const m={buildEnv:p.F3,distMethod:p.Xs,version:p.xv,originTime:h.WN},v={consented:!1},y={activatedFeatures:void 0,appMetadata:{},configured:!1,get consented(){return this.session?.state?.consent||v.consented},set consented(e){v.consented=e},customTransaction:void 0,denyList:[],disabled:!1,drainRegistry:new Map,harvester:void 0,isolatedBacklog:!1,isRecording:!1,loaderType:void 0,maxBytes:3e4,obfuscator:void 0,onerror:void 0,ptid:void 0,releaseIds:{},session:void 0,timeKeeper:void 0,registeredEntities:[],jsAttributesMetadata:{bytes:0},get harvestCount(){return++g}},b=e=>{const t=(0,u.a)(e,y),r=Object.keys(m).reduce((e,t)=>(e[t]={value:m[t],writable:!1,configurable:!0,enumerable:!0},e),{});return Object.defineProperties(t,r)},w=e=>{const t=e.startsWith("http");e+="/",r.p=t?e:"https://"+e};var R=r(7836),E=r(3241);const T={accountID:void 0,trustKey:void 0,agentID:void 0,licenseKey:void 0,applicationID:void 0,xpid:void 0},A=e=>(0,u.a)(e,T);function x(e,t={},r,o){let{init:a,info:c,loader_config:d,runtime:u={},exposed:l=!0}=t;if(!c){const e=(0,n.pV)();a=e.init,c=e.info,d=e.loader_config}e.init=f(a||{}),e.loader_config=A(d||{}),c.jsAttributes??={},h.bv&&(c.jsAttributes.isWorker=!0),e.info=(0,s.D)(c);const p=e.init;e.runtime??=b(u),p.proxy.assets&&w(p.proxy.assets),e.runtime.configured||(Object.defineProperty(e,"beacons",{get:()=>[e.info.beacon,e.info.errorBeacon,e.init.proxy.assets,e.init.proxy.beacon].filter(Boolean)}),Object.defineProperty(e.runtime,"denyList",{get:()=>[...e.init.ajax.deny_list||[],...e.init.ajax.block_internal?e.beacons:[]]}),e.runtime.ptid=e.agentIdentifier,function(e){const t=(0,n.pV)();Object.getOwnPropertyNames(i.W.prototype).forEach(r=>{const n=i.W.prototype[r];if("function"!=typeof n||"constructor"===n)return;let s=t[r];e[r]&&!1!==e.exposed&&"micro-agent"!==e.runtime?.loaderType&&(t[r]=(...t)=>{const n=e[r](...t);return s?s(...t):n})})}(e),e.runtime.loaderType=r,e.ee=R.ee.get(e.agentIdentifier),e.exposed=l,(0,E.W)({drained:!!e.runtime.activatedFeatures,type:"lifecycle",name:"initialize",feature:void 0,data:e.config}),e.runtime.configured=!0)}},5270:(e,t,r)=>{"use strict";r.d(t,{Aw:()=>o,SR:()=>s,rF:()=>a});var n=r(384),i=r(7767);function s(e){return!!(0,n.dV)().o.MO&&(0,i.V)(e)&&!0===e?.session_trace.enabled}function o(e){return!0===e?.session_replay.preload&&s(e)}function a(e,t){try{if("string"==typeof t?.type){if("password"===t.type.toLowerCase())return"*".repeat(e?.length||0);if(void 0!==t?.dataset?.nrUnmask||t?.classList?.contains("nr-unmask"))return e}}catch(e){}return"string"==typeof e?e.replace(/[\S]/g,"*"):"*".repeat(e?.length||0)}},5289:(e,t,r)=>{"use strict";r.d(t,{GG:()=>o,Qr:()=>c,sB:()=>a});var n=r(3878),i=r(6389);function s(){return"undefined"==typeof document||"complete"===document.readyState}function o(e,t){if(s())return e();const r=(0,i.J)(e),o=setInterval(()=>{s()&&(clearInterval(o),r())},500);(0,n.sp)("load",r,t)}function a(e){if(s())return e();(0,n.DD)("DOMContentLoaded",e)}function c(e){if(s())return e();(0,n.sp)("popstate",e)}},5607:(e,t,r)=>{"use strict";r.d(t,{W:()=>n});const n=(0,r(9566).bz)()},5732:(e,t,r)=>{"use strict";r.d(t,{$5:()=>u,B5:()=>d,Ms:()=>s,Ux:()=>a,YA:()=>c,fQ:()=>i,yx:()=>o});var n=r(7508);const i={MFE:"MFE",BA:"BA"};function s(e,t){if(!e||!t?.init.api.register.enabled)return[];const r=t.runtime.registeredEntities;return r?.filter(t=>String(t.metadata.target.id)===String(e)).map(e=>e.metadata.target)||[]}function o(e,t){if(!e||!t?.init.api.register.enabled)return[];const r=t.runtime.registeredEntities;return r?.filter(t=>t.metadata.timings?.asset?.endsWith(e)).map(e=>e.metadata.target)||[]}function a(e,t){if(!l(t))return{};const r=t.agentRef.runtime.appMetadata.agents[0].entityGuid;return e?e.attributes:{"entity.guid":r,appId:t.agentRef.info.applicationID}}function c(e,t){return d(e,t)?{"child.id":e.id,"child.type":e.type,...a(void 0,t)}:{}}function d(e,t){return!!e&&!!l(t)&&t.agentRef.init.api.register.duplicate_data_to_container}function u(e){if(!e?.init.api.register.enabled)return[void 0];const t=[];try{var r=(0,n.AZ)((0,n.QL)());let i=r.length-1;for(;r[i];)t.push(...o(r[i--],e))}catch(e){}return t.length||t.push(void 0),t}function l(e){return 2===e?.harvestEndpointVersion}},6154:(e,t,r)=>{"use strict";r.d(t,{OF:()=>d,RI:()=>i,WN:()=>f,bv:()=>s,gm:()=>o,lR:()=>l,m:()=>c,mw:()=>a,sb:()=>u,zk:()=>h});var n=r(1863);const i="undefined"!=typeof window&&!!window.document,s="undefined"!=typeof WorkerGlobalScope&&("undefined"!=typeof self&&self instanceof WorkerGlobalScope&&self.navigator instanceof WorkerNavigator||"undefined"!=typeof globalThis&&globalThis instanceof WorkerGlobalScope&&globalThis.navigator instanceof WorkerNavigator),o=i?window:"undefined"!=typeof WorkerGlobalScope&&("undefined"!=typeof self&&self instanceof WorkerGlobalScope&&self||"undefined"!=typeof globalThis&&globalThis instanceof WorkerGlobalScope&&globalThis),a=Boolean("hidden"===o?.document?.visibilityState),c=""+o?.location,d=/iPad|iPhone|iPod/.test(o.navigator?.userAgent),u=d&&"undefined"==typeof SharedWorker,l=(()=>{const e=o.navigator?.userAgent?.match(/Firefox[/\s](\d+\.\d+)/);return Array.isArray(e)&&e.length>=2?+e[1]:0})(),f=Date.now()-(0,n.t)(),h=()=>{const e=o?.performance?.getEntriesByType?.("navigation")?.[0];if(e&&e.responseStart>0&&e.responseStart<o.performance.now())return e}},6344:(e,t,r)=>{"use strict";r.d(t,{BB:()=>u,Qb:()=>l,TZ:()=>i,Ug:()=>o,Vh:()=>s,_s:()=>a,bc:()=>d,yP:()=>c});var n=r(2614);const i=r(860).K7.sessionReplay,s="errorDuringReplay",o=.12,a={DomContentLoaded:0,Load:1,FullSnapshot:2,IncrementalSnapshot:3,Meta:4,Custom:5},c={[n.g.ERROR]:15e3,[n.g.FULL]:3e5,[n.g.OFF]:0},d={RESET:{message:"Session was reset",sm:"Reset"},IMPORT:{message:"Recorder failed to import",sm:"Import"},TOO_MANY:{message:"429: Too Many Requests",sm:"Too-Many"},TOO_BIG:{message:"Payload was too large",sm:"Too-Big"},CROSS_TAB:{message:"Session Entity was set to OFF on another tab",sm:"Cross-Tab"},ENTITLEMENTS:{message:"Session Replay is not allowed and will not be started",sm:"Entitlement"}},u=5e3,l={API:"api",RESUME:"resume",SWITCH_TO_FULL:"switchToFull",INITIALIZE:"initialize",PRELOAD:"preload"}},6389:(e,t,r)=>{"use strict";function n(e,t=500,r={}){const n=r?.leading||!1;let i;return(...r)=>{n&&void 0===i&&(e.apply(this,r),i=setTimeout(()=>{i=clearTimeout(i)},t)),n||(clearTimeout(i),i=setTimeout(()=>{e.apply(this,r)},t))}}function i(e){let t=!1;return(...r)=>{t||(t=!0,e.apply(this,r))}}r.d(t,{J:()=>i,s:()=>n})},6630:(e,t,r)=>{"use strict";r.d(t,{T:()=>n});const n=r(860).K7.pageViewEvent},6774:(e,t,r)=>{"use strict";r.d(t,{T:()=>n});const n=r(860).K7.jserrors},7295:(e,t,r)=>{"use strict";r.d(t,{Xv:()=>o,gX:()=>i,iW:()=>s});var n=[];function i(e){if(!e||s(e))return!1;if(0===n.length)return!0;if("*"===n[0].hostname)return!1;for(var t=0;t<n.length;t++){var r=n[t];if(r.hostname.test(e.hostname)&&r.pathname.test(e.pathname))return!1}return!0}function s(e){return void 0===e.hostname}function o(e){if(n=[],e&&e.length)for(var t=0;t<e.length;t++){let r=e[t];if(!r)continue;if("*"===r)return void(n=[{hostname:"*"}]);0===r.indexOf("http://")?r=r.substring(7):0===r.indexOf("https://")&&(r=r.substring(8));const i=r.indexOf("/");let s,o;i>0?(s=r.substring(0,i),o=r.substring(i)):(s=r,o="*");let[c]=s.split(":");n.push({hostname:a(c),pathname:a(o,!0)})}}function a(e,t=!1){const r=e.replace(/[.+?^${}()|[\]\\]/g,e=>"\\"+e).replace(/\*/g,".*?");return new RegExp((t?"^":"")+r+"$")}},7485:(e,t,r)=>{"use strict";r.d(t,{D:()=>i});var n=r(6154);function i(e){if(0===(e||"").indexOf("data:"))return{protocol:"data"};try{const t=new URL(e,location.href),r={port:t.port,hostname:t.hostname,pathname:t.pathname,search:t.search,protocol:t.protocol.slice(0,t.protocol.indexOf(":")),sameOrigin:t.protocol===n.gm?.location?.protocol&&t.host===n.gm?.location?.host};return r.port&&""!==r.port||("http:"===t.protocol&&(r.port="80"),"https:"===t.protocol&&(r.port="443")),r.pathname&&""!==r.pathname?r.pathname.startsWith("/")||(r.pathname="/".concat(r.pathname)):r.pathname="/",r}catch(e){return{}}}},7508:(e,t,r)=>{"use strict";r.d(t,{AZ:()=>g,Qr:()=>b,QL:()=>m});var n=r(6154),i=r(1863),s=r(9119),o=r(7866);class a{dom=new c;performance=new c;constructor(e){this.url=e}get script(){const e=Math.max(this.dom.start,this.performance.end);return{start:e,end:Math.max(this.dom.end,this.performance.end,e)}}}class c{start=0;end=0;value=void 0}let d;try{d=g(m())[0]}catch(e){d=g(e)[0]}const u=e=>"script"===e.initiatorType||["link","fetch"].includes(e.initiatorType)&&e.name.endsWith(".js"),l=new Map;let f=[];function h(e){return l.get(e)}function p(e){const t=h(e);if(t)return t;const r=new a(e);if(l.set(e,r),l.size>1e3){const e=l.keys().next().value;l.delete(e)}return r}if(n.gm.MutationObserver&&n.gm.document){new MutationObserver(e=>{e.forEach(e=>{e.addedNodes.forEach(e=>{if("SCRIPT"===e.nodeName&&e.src){const t=p((0,s.L)(e.src));t.dom.start=(0,i.t)(),t.dom.value=e;const r=()=>{t.dom.end=(0,i.t)()};["load","error"].forEach(t=>e.addEventListener(t,r,{once:!0}))}})})}).observe(n.gm.document,{childList:!0,subtree:!0})}if(n.gm.PerformanceObserver?.supportedEntryTypes.includes("resource")){new PerformanceObserver(e=>{e.getEntries().filter(u).forEach(e=>{const t=p((0,s.L)(e.name));t.performance.start=Math.floor(e.startTime),t.performance.end=Math.floor(e.responseEnd),t.performance.value=e;const r=[];f.forEach(({test:t,addedAt:n},s)=>{(t(e)||(0,i.t)()-n>1e4)&&r.push(s)}),f=f.filter((e,t)=>!r.includes(t))})}).observe({type:"resource",buffered:!0})}function g(e){if(!e||"string"!=typeof e)return[];const t=new Set,r=e.split("\n");for(const e of r){const r=e.match(o.cn)||e.match(o.hB)||e.match(o.fL);if(r&&r[2])t.add((0,s.L)(r[2]));else{const r=e.match(/\(([^)]+\.js):\d+:\d+\)/)||e.match(/^\s+at\s+([^\s(]+\.js):\d+:\d+/);r&&r[1]&&t.add((0,s.L)(r[1]))}}return[...t]}function m(){let e;try{const t=Error.stackTraceLimit;Error.stackTraceLimit=50,e=(new Error).stack,Error.stackTraceLimit=t}catch(t){e=(new Error).stack}return e}function v(e,t){return(0,s.L)(e.name)===t}function y(e,t){e.fetchStart=Math.floor(t.startTime),e.fetchEnd=Math.floor(t.responseEnd),e.asset=t.name,e.type=t.initiatorType}function b(){const e={registeredAt:(0,i.t)(),reportedAt:void 0,fetchStart:0,fetchEnd:0,scriptStart:0,scriptEnd:0,asset:void 0,type:"unknown"},t=m();if(!t)return e;const r=n.gm.performance?.getEntriesByType("navigation")?.[0]?.name||"";try{const o=g(t),a=(o.length>1?o.filter(e=>d!==e):o)[0];if(!a)return e;if(r.includes(a))return e.asset=(0,s.L)(r),e.type="inline",e;e.correlation=h(a);const c=e.correlation?.performance.value||performance.getEntriesByType("resource").find(e=>v(e,a));c?y(e,c):function(e){if(!e||!n.gm.document)return!1;try{const t=n.gm.document.querySelectorAll('link[rel="preload"][as="script"]');for(const r of t)if((0,s.L)(r.href)===e)return!0}catch(e){}return!1}(a)&&(e.asset=a,e.type="preload",f.push({addedAt:(0,i.t)(),test:t=>!!v(t,a)&&(y(e,t),!0)})),Object.defineProperty(e,"scriptStart",{get:()=>e.correlation?.script.start||e.fetchEnd}),Object.defineProperty(e,"scriptEnd",{get:()=>e.correlation?.script.end||e.registeredAt})}catch(e){}return e}},7699:(e,t,r)=>{"use strict";r.d(t,{It:()=>s,KC:()=>a,No:()=>i,qh:()=>o});var n=r(860);const i=16e3,s=1e6,o="SESSION_ERROR",a={[n.K7.logging]:!0,[n.K7.genericEvents]:!0,[n.K7.jserrors]:!0,[n.K7.ajax]:!0}},7767:(e,t,r)=>{"use strict";r.d(t,{V:()=>i});var n=r(6154);const i=e=>n.RI&&!0===e?.privacy.cookies_enabled},7836:(e,t,r)=>{"use strict";r.d(t,{P:()=>a,ee:()=>c});var n=r(384),i=r(8990),s=r(2646),o=r(5607);const a="nr@context:".concat(o.W),c=function e(t,r){var n={},o={},u={},l=!1;try{l=16===r.length&&d.initializedAgents?.[r]?.runtime.isolatedBacklog}catch(e){}var f={on:p,addEventListener:p,removeEventListener:function(e,t){var r=n[e];if(!r)return;for(var i=0;i<r.length;i++)r[i]===t&&r.splice(i,1)},emit:function(e,r,n,i,s){!1!==s&&(s=!0);if(c.aborted&&!i)return;t&&s&&t.emit(e,r,n);var a=h(n);g(e).forEach(e=>{e.apply(a,r)});var d=v()[o[e]];d&&d.push([f,e,r,a]);return a},get:m,listeners:g,context:h,buffer:function(e,t){const r=v();if(t=t||"feature",f.aborted)return;Object.entries(e||{}).forEach(([e,n])=>{o[n]=t,t in r||(r[t]=[])})},abort:function(){f._aborted=!0,Object.keys(f.backlog).forEach(e=>{delete f.backlog[e]})},isBuffering:function(e){return!!v()[o[e]]},debugId:r,backlog:l?{}:t&&"object"==typeof t.backlog?t.backlog:{},isolatedBacklog:l};return Object.defineProperty(f,"aborted",{get:()=>{let e=f._aborted||!1;return e||(t&&(e=t.aborted),e)}}),f;function h(e){return e&&e instanceof s.y?e:e?(0,i.I)(e,a,()=>new s.y(a)):new s.y(a)}function p(e,t){n[e]=g(e).concat(t)}function g(e){return n[e]||[]}function m(t){return u[t]=u[t]||e(f,t)}function v(){return f.backlog}}(void 0,"globalEE"),d=(0,n.Zm)();d.ee||(d.ee=c)},7866:(e,t,r)=>{"use strict";r.d(t,{Nc:()=>s,cn:()=>a,fL:()=>i,h3:()=>n,hB:()=>o});const n=/function (.+?)\s*\(/,i=/^\s*at .+ \(eval at \S+ \((?:(?:file|http|https):[^)]+)?\)(?:, [^:]*:\d+:\d+)?\)$/i,s=/^\s*at Function code \(Function code:\d+:\d+\)\s*/i,o=/^\s*at (?:((?:\[object object\])?(?:[^(]*\([^)]*\))*[^()]*(?: \[as \S+\])?) )?\(?((?:file|http|https|chrome-extension):.*?)?:(\d+)(?::(\d+))?\)?\s*$/i,a=/^\s*(?:([^@]*)(?:\(.*?\))?@)?((?:file|http|https|chrome|safari-extension).*?):(\d+)(?::(\d+))?\s*$/i},8122:(e,t,r)=>{"use strict";r.d(t,{a:()=>i});var n=r(944);function i(e,t){try{if(!e||"object"!=typeof e)return(0,n.R)(3);if(!t||"object"!=typeof t)return(0,n.R)(4);const r=Object.create(Object.getPrototypeOf(t),Object.getOwnPropertyDescriptors(t)),s=0===Object.keys(r).length?e:r;for(let o in s)if(void 0!==e[o])try{if(null===e[o]){r[o]=null;continue}Array.isArray(e[o])&&Array.isArray(t[o])?r[o]=Array.from(new Set([...e[o],...t[o]])):e[o]instanceof Map||e[o]instanceof Set||e[o]instanceof Date||e[o]instanceof RegExp?r[o]=e[o]:"object"==typeof e[o]&&"object"==typeof t[o]?r[o]=i(e[o],t[o]):r[o]=e[o]}catch(e){r[o]||(0,n.R)(1,e)}return r}catch(e){(0,n.R)(2,e)}}},8139:(e,t,r)=>{"use strict";r.d(t,{u:()=>f});var n=r(7836),i=r(3434),s=r(8990),o=r(6154);const a={},c=o.gm.XMLHttpRequest,d="addEventListener",u="removeEventListener",l="nr@wrapped:".concat(n.P);function f(e){var t=function(e){return(e||n.ee).get("events")}(e);if(a[t.debugId]++)return t;a[t.debugId]=1;var r=(0,i.YM)(t,!0);function f(e){r.inPlace(e,[d,u],"-",p)}function p(e,t){return e[1]}return"getPrototypeOf"in Object&&(o.RI&&h(document,f),c&&h(c.prototype,f),h(o.gm,f)),t.on(d+"-start",function(e,t){var n=e[1];if(null!==n&&("function"==typeof n||"object"==typeof n)&&"newrelic"!==e[0]){var i=(0,s.I)(n,l,function(){var e={object:function(){if("function"!=typeof n.handleEvent)return;return n.handleEvent.apply(n,arguments)},function:n}[typeof n];return e?r(e,"fn-",null,e.name||"anonymous"):n});this.wrapped=e[1]=i}}),t.on(u+"-start",function(e){e[1]=this.wrapped||e[1]}),t}function h(e,t,...r){let n=e;for(;"object"==typeof n&&!Object.prototype.hasOwnProperty.call(n,d);)n=Object.getPrototypeOf(n);n&&t(n,...r)}},8362:(e,t,r)=>{"use strict";r.d(t,{d:()=>s});var n=r(9566),i=r(1741);class s extends i.W{agentIdentifier=(0,n.LA)(16)}},8374:(e,t,r)=>{r.nc=(()=>{try{return document?.currentScript?.nonce}catch(e){}return""})()},8990:(e,t,r)=>{"use strict";r.d(t,{I:()=>i});var n=Object.prototype.hasOwnProperty;function i(e,t,r){if(n.call(e,t))return e[t];var i=r();if(Object.defineProperty&&Object.keys)try{return Object.defineProperty(e,t,{value:i,writable:!0,enumerable:!1}),i}catch(e){}return e[t]=i,i}},9119:(e,t,r)=>{"use strict";r.d(t,{L:()=>s});var n=/([^?#]*)[^#]*(#[^?]*|$).*/,i=/([^?#]*)().*/;function s(e,t){return e?e.replace(t?n:i,"$1$2"):e}},9300:(e,t,r)=>{"use strict";r.d(t,{T:()=>n,f:()=>i});const n=r(860).K7.ajax,i="ajaxRequest.id"},9324:(e,t,r)=>{"use strict";r.d(t,{AJ:()=>o,F3:()=>i,Xs:()=>s,Yq:()=>a,xv:()=>n});const n="1.314.0",i="PROD",s="CDN",o="@newrelic/rrweb",a="1.1.0"},9566:(e,t,r)=>{"use strict";r.d(t,{LA:()=>a,ZF:()=>c,bz:()=>o,el:()=>d});var n=r(6154);const i="xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx";function s(e,t){return e?15&e[t]:16*Math.random()|0}function o(){const e=n.gm?.crypto||n.gm?.msCrypto;let t,r=0;return e&&e.getRandomValues&&(t=e.getRandomValues(new Uint8Array(30))),i.split("").map(e=>"x"===e?s(t,r++).toString(16):"y"===e?(3&s()|8).toString(16):e).join("")}function a(e){const t=n.gm?.crypto||n.gm?.msCrypto;let r,i=0;t&&t.getRandomValues&&(r=t.getRandomValues(new Uint8Array(e)));const o=[];for(var a=0;a<e;a++)o.push(s(r,i++).toString(16));return o.join("")}function c(){return a(16)}function d(){return a(32)}},9908:(e,t,r)=>{"use strict";r.d(t,{d:()=>n,p:()=>i});var n=r(7836).ee.get("handle");function i(e,t,r,i,s){s?(s.buffer([e],i),s.emit(e,t,r)):(n.buffer([e],i),n.emit(e,t,r))}}},n={};function i(e){var t=n[e];if(void 0!==t)return t.exports;var s=n[e]={exports:{}};return r[e](s,s.exports,i),s.exports}i.m=r,i.d=(e,t)=>{for(var r in t)i.o(t,r)&&!i.o(e,r)&&Object.defineProperty(e,r,{enumerable:!0,get:t[r]})},i.f={},i.e=e=>Promise.all(Object.keys(i.f).reduce((t,r)=>(i.f[r](e,t),t),[])),i.u=e=>({212:"nr-spa-compressor",249:"nr-spa-recorder",478:"nr-spa"}[e]+"-1.314.0.min.js"),i.o=(e,t)=>Object.prototype.hasOwnProperty.call(e,t),e={},t="NRBA-1.314.0.PROD:",i.l=(r,n,s,o)=>{if(e[r])e[r].push(n);else{var a,c;if(void 0!==s)for(var d=document.getElementsByTagName("script"),u=0;u<d.length;u++){var l=d[u];if(l.getAttribute("src")==r||l.getAttribute("data-webpack")==t+s){a=l;break}}if(!a){c=!0;var f={478:"sha512-mgj9qD5BdNJpEJCPmd1OfPzfnL/HRJ64OPYq/GZQxKc0756ytnQixcg8IA8iZWe/PRWN4DuGAnX82J2lscbiIw==",249:"sha512-fUJXIKjSt6UL9n1UZzNMzGGL3CxnBT/AgtoTat0/ONINbEB2PoiO+yK54DPYDwzpXDBddJHvs2BrYFj1GBpo2Q==",212:"sha512-vjoeBTgR0ONx6WcgBdfuPgcta1834zMPDhLL05C9c/pB+/xAuxL4BJMu9yeXYh+oIV7oIn6qTkDC7rBcZIDiHg=="};(a=document.createElement("script")).charset="utf-8",i.nc&&a.setAttribute("nonce",i.nc),a.setAttribute("data-webpack",t+s),a.src=r,0!==a.src.indexOf(window.location.origin+"/")&&(a.crossOrigin="anonymous"),f[o]&&(a.integrity=f[o])}e[r]=[n];var h=(t,n)=>{a.onerror=a.onload=null,clearTimeout(p);var i=e[r];if(delete e[r],a.parentNode&&a.parentNode.removeChild(a),i&&i.forEach(e=>e(n)),t)return t(n)},p=setTimeout(h.bind(null,void 0,{type:"timeout",target:a}),12e4);a.onerror=h.bind(null,a.onerror),a.onload=h.bind(null,a.onload),c&&document.head.appendChild(a)}},i.r=e=>{"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},i.p="https://js-agent.newrelic.com/",(()=>{var e={38:0,788:0};i.f.j=(t,r)=>{var n=i.o(e,t)?e[t]:void 0;if(0!==n)if(n)r.push(n[2]);else{var s=new Promise((r,i)=>n=e[t]=[r,i]);r.push(n[2]=s);var o=i.p+i.u(t),a=new Error;i.l(o,r=>{if(i.o(e,t)&&(0!==(n=e[t])&&(e[t]=void 0),n)){var s=r&&("load"===r.type?"missing":r.type),o=r&&r.target&&r.target.src;a.message="Loading chunk "+t+" failed: ("+s+": "+o+")",a.name="ChunkLoadError",a.type=s,a.request=o,n[1](a)}},"chunk-"+t,t)}};var t=(t,r)=>{var n,s,[o,a,c]=r,d=0;if(o.some(t=>0!==e[t])){for(n in a)i.o(a,n)&&(i.m[n]=a[n]);if(c)c(i)}for(t&&t(r);d<o.length;d++)s=o[d],i.o(e,s)&&e[s]&&e[s][0](),e[s]=0},r=self["webpackChunk:NRBA-1.314.0.PROD"]=self["webpackChunk:NRBA-1.314.0.PROD"]||[];r.forEach(t.bind(null,0)),r.push=t.bind(null,r.push.bind(r))})(),(()=>{"use strict";i(8374);var e=i(8362),t=i(860);const r=Object.values(t.K7);var n=i(5205);var s=i(9908),o=i(1863),a=i(4261),c=i(1738);var d=i(1687),u=i(4234),l=i(5289),f=i(6154),h=i(944),p=i(5270),g=i(7767),m=i(6389),v=i(7699);class y extends u.W{constructor(e,t){super(e,t),this.abortHandler=void 0,this.featAggregate=void 0,this.loadedSuccessfully=void 0,this.onAggregateImported=new Promise(e=>{this.loadedSuccessfully=e}),this.deferred=Promise.resolve(),!1===e.init[this.featureName].autoStart?this.deferred=new Promise((t,r)=>{this.ee.on("manual-start-all",(0,m.J)(()=>{(0,d.Ak)(e,this.featureName),t()}))}):(0,d.Ak)(e,t)}importAggregator(e,t,r={}){if(this.featAggregate)return;const n=async()=>{let n;await this.deferred;try{if((0,g.V)(e.init)){const{setupAgentSession:t}=await i.e(478).then(i.bind(i,8766));n=t(e)}}catch(e){(0,h.R)(20,e),this.ee.emit("internal-error",[e]),(0,s.p)(v.qh,[e],void 0,this.featureName,this.ee)}try{if(!this.#t(this.featureName,n,e.init))return(0,d.Ze)(this.agentRef,this.featureName),void this.loadedSuccessfully(!1);const{Aggregate:i}=await t();this.featAggregate=new i(e,r),e.runtime.harvester.initializedAggregates.push(this.featAggregate),this.loadedSuccessfully(!0)}catch(e){(0,h.R)(34,e),this.abortHandler?.(),(0,d.Ze)(this.agentRef,this.featureName,!0),this.loadedSuccessfully(!1),this.ee&&this.ee.abort()}};f.RI?(0,l.GG)(()=>n(),!0):n()}#t(e,r,n){if(this.blocked)return!1;switch(e){case t.K7.sessionReplay:return(0,p.SR)(n)&&!!r;case t.K7.sessionTrace:return!!r;default:return!0}}}var b=i(6630),w=i(2614),R=i(3241);class E extends y{static featureName=b.T;constructor(e){var t;super(e,b.T),this.setupInspectionEvents(),t=e,(0,c.Y)(a.Fw,function(e,r){"string"==typeof e&&("/"!==e.charAt(0)&&(e="/"+e),t.runtime.customTransaction=(r||"http://custom.transaction")+e,(0,s.p)(a.Pl+a.Fw,[(0,o.t)()],void 0,void 0,t.ee))},t),this.importAggregator(e,()=>i.e(478).then(i.bind(i,5839)))}setupInspectionEvents(){const e=(e,t)=>{e&&(0,R.W)({timeStamp:e.timeStamp,loaded:"complete"===e.target.readyState,type:"window",name:t,data:e.target.location+""})};(0,l.sB)(t=>{e(t,"DOMContentLoaded")}),(0,l.GG)(t=>{e(t,"load")}),(0,l.Qr)(t=>{e(t,"navigate")}),this.ee.on(w.tS.UPDATE,(e,t)=>{(0,R.W)({type:"lifecycle",name:"session",data:t})})}}var T=i(384);class A extends e.d{constructor(e){var t;(super(),f.gm)?(this.features={},(0,T.bQ)(this.agentIdentifier,this),this.desiredFeatures=new Set(e.features||[]),this.desiredFeatures.add(E),(0,n.j)(this,e,e.loaderType||"agent"),t=this,(0,c.Y)(a.cD,function(e,r,n=!1){if("string"==typeof e){if(["string","number","boolean"].includes(typeof r)||null===r)return(0,c.U)(t,e,r,a.cD,n);(0,h.R)(40,typeof r)}else(0,h.R)(39,typeof e)},t),function(e){(0,c.Y)(a.Dl,function(t,r=!1){if("string"!=typeof t&&null!==t)return void(0,h.R)(41,typeof t);const n=e.info.jsAttributes["enduser.id"];r&&null!=n&&n!==t?(0,s.p)(a.Pl+"setUserIdAndResetSession",[t],void 0,"session",e.ee):(0,c.U)(e,"enduser.id",t,a.Dl,!0)},e)}(this),function(e){(0,c.Y)(a.nb,function(t){if("string"==typeof t||null===t)return(0,c.U)(e,"application.version",t,a.nb,!1);(0,h.R)(42,typeof t)},e)}(this),function(e){(0,c.Y)(a.d3,function(){e.ee.emit("manual-start-all")},e)}(this),function(e){(0,c.Y)(a.Pv,function(t=!0){if("boolean"==typeof t){if((0,s.p)(a.Pl+a.Pv,[t],void 0,"session",e.ee),e.runtime.consented=t,t){const t=e.features.page_view_event;t.onAggregateImported.then(e=>{const r=t.featAggregate;e&&!r.sentRum&&r.sendRum()})}}else(0,h.R)(65,typeof t)},e)}(this),this.run()):(0,h.R)(21)}get config(){return{info:this.info,init:this.init,loader_config:this.loader_config,runtime:this.runtime}}get api(){return this}run(){try{const e=function(e){const t={};return r.forEach(r=>{t[r]=!!e[r]?.enabled}),t}(this.init),n=[...this.desiredFeatures];n.sort((e,r)=>t.P3[e.featureName]-t.P3[r.featureName]),n.forEach(r=>{if(!e[r.featureName]&&r.featureName!==t.K7.pageViewEvent)return;const n=function(e){switch(e){case t.K7.ajax:return[t.K7.jserrors];case t.K7.sessionTrace:return[t.K7.ajax,t.K7.pageViewEvent];case t.K7.sessionReplay:return[t.K7.sessionTrace];case t.K7.pageViewTiming:return[t.K7.pageViewEvent];default:return[]}}(r.featureName).filter(e=>!(e in this.features));n.length>0&&(0,h.R)(36,{targetFeature:r.featureName,missingDependencies:n}),this.features[r.featureName]=new r(this)})}catch(e){(0,h.R)(22,e);for(const e in this.features)this.features[e].abortHandler?.();const t=(0,T.Zm)();delete t.initializedAgents[this.agentIdentifier]?.features,delete this.sharedAggregator;return t.ee.get(this.agentIdentifier).abort(),!1}}}var x=i(2843),S=i(782);class _ extends y{static featureName=S.T;constructor(e){super(e,S.T),f.RI&&((0,x.u)(()=>(0,s.p)("docHidden",[(0,o.t)()],void 0,S.T,this.ee),!0),(0,x.G)(()=>(0,s.p)("winPagehide",[(0,o.t)()],void 0,S.T,this.ee)),this.importAggregator(e,()=>i.e(478).then(i.bind(i,9917))))}}var O=i(3969);class P extends y{static featureName=O.TZ;constructor(e){super(e,O.TZ),this.importAggregator(e,()=>i.e(478).then(i.bind(i,6555)))}}var k=i(6774),N=i(3878),j=i(3304);class D{constructor(e,t,r,n,i){this.name="UncaughtError",this.message="string"==typeof e?e:(0,j.A)(e),this.sourceURL=t,this.line=r,this.column=n,this.__newrelic=i}}function C(e){return M(e)?e:new D(void 0!==e?.message?e.message:e,e?.filename||e?.sourceURL,e?.lineno||e?.line,e?.colno||e?.col,e?.__newrelic,e?.cause)}function L(e){const t="Unhandled Promise Rejection: ";if(!e?.reason)return;if(M(e.reason)){try{e.reason.message.startsWith(t)||(e.reason.message=t+e.reason.message)}catch(e){}return C(e.reason)}const r=C(e.reason);return(r.message||"").startsWith(t)||(r.message=t+r.message),r}function I(e){if(e.error instanceof SyntaxError&&!/:\d+$/.test(e.error.stack?.trim())){const t=new D(e.message,e.filename,e.lineno,e.colno,e.error.__newrelic,e.cause);return t.name=SyntaxError.name,t}return M(e.error)?e.error:C(e)}function M(e){return e instanceof Error&&!!e.stack}function B(e,r,n,i,a=(0,o.t)()){"string"==typeof e&&(e=new Error(e)),(0,s.p)("err",[e,a,!1,r,n.runtime.isRecording,void 0,i],void 0,t.K7.jserrors,n.ee),(0,s.p)("uaErr",[],void 0,t.K7.genericEvents,n.ee)}var H=i(5732),K=i(993),W=i(3785);function F(e,{customAttributes:t={},level:r=K.p_.INFO}={},n,i,s=(0,o.t)()){(0,W.R)(n.ee,e,t,r,!1,i,s)}function U(e,r,n,i,c=(0,o.t)()){(0,s.p)(a.Pl+a.hG,[c,e,r,i],void 0,t.K7.genericEvents,n.ee)}function V(e,r,n,i,c=(0,o.t)()){const{start:d,end:u,customAttributes:l}=r||{},f={customAttributes:l||{}};if("object"!=typeof f.customAttributes||"string"!=typeof e||0===e.length)return void(0,h.R)(57);const p=(e,t)=>null==e?t:"number"==typeof e?e:e instanceof PerformanceMark?e.startTime:Number.NaN;if(f.start=p(d,0),f.end=p(u,c),Number.isNaN(f.start)||Number.isNaN(f.end))(0,h.R)(57);else{if(f.duration=f.end-f.start,!(f.duration<0))return(0,s.p)(a.Pl+a.V1,[f,e,i],void 0,t.K7.genericEvents,n.ee),f;(0,h.R)(58)}}function z(e,r={},n,i,c=(0,o.t)()){(0,s.p)(a.Pl+a.fF,[c,e,r,i],void 0,t.K7.genericEvents,n.ee)}var G=i(7508),Y=i(9566);const Z=["name","id","type"],q=new Map([[U,"addPageAction"],[F,"log"],[V,"measure"],[B,"noticeError"],[z,"recordCustomEvent"]]),X={experimental:(0,m.J)(()=>(0,h.R)(54,"newrelic.register")),disabled:(0,m.J)(()=>(0,h.R)(55)),invalidTarget:(0,m.J)(e=>(0,h.R)(48,e)),deregistered:(0,m.J)(()=>(0,h.R)(68))};function Q(e){(0,c.Y)(a.eY,function(t){return J(e,t)},e)}function J(e,r){X.experimental(),r||={},r.instance=(0,Y.LA)(8),r.type=H.fQ.MFE,r.licenseKey||=e.info.licenseKey,r.blocked=!1,("object"!=typeof r.tags||null===r.tags||Array.isArray(r.tags))&&(r.tags={}),r.parent??={get id(){return e.runtime.appMetadata.agents[0].entityGuid},type:H.fQ.BA};const n=(0,G.Qr)(),i={};Object.prototype.hasOwnProperty.call(r,"attributes")||Object.defineProperty(r,"attributes",{get:()=>({...i,"source.id":r.id,"source.name":r.name,"source.type":r.type,"parent.type":r.parent?.type||H.fQ.BA,"parent.id":r.parent?.id})}),Object.entries(r.tags).forEach(([e,t])=>{Z.includes(e)||(i["source.".concat(e)]=t)});let a=()=>{};const c=e.runtime.registeredEntities,d=e=>{r.blocked=!0,a=e};function u(e){return"string"==typeof e&&!!e.trim()&&e.trim().length<501}e.init.api.register.enabled||d(X.disabled),u(r.id)&&u(r.name)||d(()=>X.invalidTarget(r));const l={addPageAction:(t,n={})=>m(U,[t,{...i,...n},e],r),deregister:()=>{p(),d(X.deregistered)},log:(t,n={})=>m(F,[t,{...n,customAttributes:{...i,...n.customAttributes||{}}},e],r),measure:(t,n={})=>m(V,[t,{...n,customAttributes:{...i,...n.customAttributes||{}}},e],r),noticeError:(t,n={})=>m(B,[t,{...i,...n},e],r),recordCustomEvent:(t,n={})=>m(z,[t,{...i,...n},e],r),setApplicationVersion:e=>g("application.version",e),setCustomAttribute:(e,t)=>g(e,t),setUserId:e=>g("enduser.id",e),metadata:{get customAttributes(){return i},target:r,timings:n}},f=()=>(r.blocked&&a(),r.blocked);function p(){if(n.reportedAt)return;n.reportedAt=(0,o.t)();const e=n.fetchEnd-n.fetchStart,t=n.scriptEnd-n.scriptStart;l.recordCustomEvent("MicroFrontEndTiming",{assetUrl:n.asset,assetType:n.type,timeAlive:n.reportedAt-n.registeredAt,timeToBeRequested:n.fetchStart,timeToExecute:t,timeToFetch:e,timeToLoad:e+t,timeToRegister:n.registeredAt})}f()||(c.push(l),(0,x.G)(p));const g=(e,t)=>{f()||(i[e]=t)},m=(r,n,i)=>{if(f()&&r!==J)return;const a=(0,o.t)(),c=q.get(r)||"unknown";(0,s.p)(O.xV,["API/register/".concat(c,"/called")],void 0,t.K7.metrics,e.ee);try{return r(...n,i,a)}catch(e){(0,h.R)(50,e)}};return l}class ee extends y{static featureName=k.T;constructor(e){var t;super(e,k.T),t=e,(0,c.Y)(a.o5,(e,r)=>B(e,r,t),t),function(e){(0,c.Y)(a.bt,function(t){e.runtime.onerror=t},e)}(e),function(e){let t=0;(0,c.Y)(a.k6,function(e,r){++t>10||(this.runtime.releaseIds[e.slice(-200)]=(""+r).slice(-200))},e)}(e),Q(e);try{this.removeOnAbort=new AbortController}catch(e){}this.ee.on("internal-error",(t,r)=>{this.abortHandler&&(0,s.p)("ierr",[C(t),(0,o.t)(),!0,{},e.runtime.isRecording,r],void 0,this.featureName,this.ee)}),f.gm.addEventListener("unhandledrejection",t=>{this.abortHandler&&(0,s.p)("err",[L(t),(0,o.t)(),!1,{unhandledPromiseRejection:1},e.runtime.isRecording],void 0,this.featureName,this.ee)},(0,N.jT)(!1,this.removeOnAbort?.signal)),f.gm.addEventListener("error",t=>{this.abortHandler&&(0,s.p)("err",[I(t),(0,o.t)(),!1,{},e.runtime.isRecording],void 0,this.featureName,this.ee)},(0,N.jT)(!1,this.removeOnAbort?.signal)),this.abortHandler=this.#r,this.importAggregator(e,()=>i.e(478).then(i.bind(i,9377)))}#r(){this.removeOnAbort?.abort(),this.abortHandler=void 0}}var te=i(8990);let re=1;function ne(e){const t=typeof e;return!e||"object"!==t&&"function"!==t?-1:e===f.gm?0:(0,te.I)(e,"nr@id",function(){return re++})}function ie(e){if("string"==typeof e&&e.length)return e.length;if("object"==typeof e){if("undefined"!=typeof ArrayBuffer&&e instanceof ArrayBuffer&&e.byteLength)return e.byteLength;if("undefined"!=typeof Blob&&e instanceof Blob&&e.size)return e.size;if(!("undefined"!=typeof FormData&&e instanceof FormData))try{return(0,j.A)(e).length}catch(e){return}}}var se=i(8139),oe=i(7836),ae=i(3434);const ce={},de=["open","send"];function ue(e,t){var r=e||oe.ee;const n=function(e){return(e||oe.ee).get("xhr")}(r);if(void 0===f.gm.XMLHttpRequest)return n;if(ce[n.debugId]++)return n;ce[n.debugId]=1,(0,se.u)(r);var i=(0,ae.YM)(n),s=f.gm.XMLHttpRequest,o=f.gm.MutationObserver,a=f.gm.Promise,c=f.gm.setInterval,d="readystatechange",u=["onload","onerror","onabort","onloadstart","onloadend","onprogress","ontimeout"],l=[],p=f.gm.XMLHttpRequest=function(e){const r=new s(e),o=n.context(r);o.targets=(0,H.$5)(t);try{n.emit("new-xhr",[r],o),r.addEventListener(d,(a=o,function(){var e=this;e.readyState>3&&!a.resolved&&(a.resolved=!0,n.emit("xhr-resolved",[],e)),i.inPlace(e,u,"fn-",w)}),(0,N.jT)(!1))}catch(e){(0,h.R)(15,e);try{n.emit("internal-error",[e])}catch(e){}}var a;return r};function g(e,t){i.inPlace(t,["onreadystatechange"],"fn-",w)}if(function(e,t){for(var r in e)t[r]=e[r]}(s,p),p.prototype=s.prototype,i.inPlace(p.prototype,de,"-xhr-",w),n.on("send-xhr-start",function(e,t){g(e,t),function(e){l.push(e),o&&(m?m.then(b):c?c(b):(v=-v,y.data=v))}(t)}),n.on("open-xhr-start",g),o){var m=a&&a.resolve();if(!c&&!a){var v=1,y=document.createTextNode(v);new o(b).observe(y,{characterData:!0})}}else r.on("fn-end",function(e){e[0]&&e[0].type===d||b()});function b(){for(var e=0;e<l.length;e++)g(0,l[e]);l.length&&(l=[])}function w(e,t){return t}return n}var le="fetch-",fe=le+"body-",he=["arrayBuffer","blob","json","text","formData"],pe=f.gm.Request,ge=f.gm.Response,me="prototype";const ve={};function ye(e,t){const r=function(e){return(e||oe.ee).get("fetch")}(e);if(!(pe&&ge&&f.gm.fetch))return r;if(ve[r.debugId]++)return r;function n(e,n,i){var s=e[n];"function"==typeof s&&(e[n]=function(){var e=[...arguments];const n={},o=(0,H.$5)(t);var a;r.emit(i+"before-start",[e],n),n[oe.P]&&n[oe.P].dt&&(a=n[oe.P].dt);var c=s.apply(this,e);return r.emit(i+"start",[e,a],c),c.then(function(e){return r.emit(i+"end",[null,e,o],c),e},function(e){throw r.emit(i+"end",[e,void 0,o],c),e})})}return ve[r.debugId]=1,he.forEach(e=>{n(pe[me],e,fe),n(ge[me],e,fe)}),n(f.gm,"fetch",le),r.on(le+"end",function(e,t,n){var i=this;if(i.targets=n||[void 0],t){var s=t.headers.get("content-length");null!==s&&(i.rxSize=s),r.emit(le+"done",[null,t],i)}else r.emit(le+"done",[e],i)}),r}var be=i(7485);class we{constructor(e){this.agentRef=e}generateTracePayload(e){const t=this.agentRef.loader_config;if(!this.shouldGenerateTrace(e)||!t)return null;var r=(t.accountID||"").toString()||null,n=(t.agentID||"").toString()||null,i=(t.trustKey||"").toString()||null;if(!r||!n)return null;var s=(0,Y.ZF)(),o=(0,Y.el)(),a=Date.now(),c={spanId:s,traceId:o,timestamp:a};return(e.sameOrigin||this.isAllowedOrigin(e)&&this.useTraceContextHeadersForCors())&&(c.traceContextParentHeader=this.generateTraceContextParentHeader(s,o),c.traceContextStateHeader=this.generateTraceContextStateHeader(s,a,r,n,i)),(e.sameOrigin&&!this.excludeNewrelicHeader()||!e.sameOrigin&&this.isAllowedOrigin(e)&&this.useNewrelicHeaderForCors())&&(c.newrelicHeader=this.generateTraceHeader(s,o,a,r,n,i)),c}generateTraceContextParentHeader(e,t){return"00-"+t+"-"+e+"-01"}generateTraceContextStateHeader(e,t,r,n,i){return i+"@nr=0-1-"+r+"-"+n+"-"+e+"----"+t}generateTraceHeader(e,t,r,n,i,s){if(!("function"==typeof f.gm?.btoa))return null;var o={v:[0,1],d:{ty:"Browser",ac:n,ap:i,id:e,tr:t,ti:r}};return s&&n!==s&&(o.d.tk=s),btoa((0,j.A)(o))}shouldGenerateTrace(e){return this.agentRef.init?.distributed_tracing?.enabled&&this.isAllowedOrigin(e)}isAllowedOrigin(e){var t=!1;const r=this.agentRef.init?.distributed_tracing;if(e.sameOrigin)t=!0;else if(r?.allowed_origins instanceof Array)for(var n=0;n<r.allowed_origins.length;n++){var i=(0,be.D)(r.allowed_origins[n]);if(e.hostname===i.hostname&&e.protocol===i.protocol&&e.port===i.port){t=!0;break}}return t}excludeNewrelicHeader(){var e=this.agentRef.init?.distributed_tracing;return!!e&&!!e.exclude_newrelic_header}useNewrelicHeaderForCors(){var e=this.agentRef.init?.distributed_tracing;return!!e&&!1!==e.cors_use_newrelic_header}useTraceContextHeadersForCors(){var e=this.agentRef.init?.distributed_tracing;return!!e&&!!e.cors_use_tracecontext_headers}}var Re=i(9300),Ee=i(7295);function Te(e){return"string"==typeof e?e:e instanceof(0,T.dV)().o.REQ?e.url:f.gm?.URL&&e instanceof URL?e.href:void 0}var Ae=["load","error","abort","timeout"],xe=Ae.length,Se=(0,T.dV)().o.REQ,_e=(0,T.dV)().o.XHR;const Oe="X-NewRelic-App-Data";class Pe extends y{static featureName=Re.T;constructor(e){super(e,Re.T),this.dt=new we(e),this.handler=(e,t,r,n)=>(0,s.p)(e,t,r,n,this.ee);try{const e={xmlhttprequest:"xhr",fetch:"fetch",beacon:"beacon"};f.gm?.performance?.getEntriesByType("resource").forEach(r=>{if(r.initiatorType in e&&0!==r.responseStatus){const n={status:r.responseStatus},i={rxSize:r.transferSize,duration:Math.floor(r.duration),cbTime:0};ke(n,r.name),this.handler("xhr",[n,i,r.startTime,r.responseEnd,e[r.initiatorType]],void 0,t.K7.ajax)}})}catch(e){}ye(this.ee,e),ue(this.ee,e),function(e,r,n,i){function a(e){var t=this;t.totalCbs=0,t.called=0,t.cbTime=0,t.end=T,t.ended=!1,t.xhrGuids={},t.lastSize=null,t.loadCaptureCalled=!1,t.params=this.params||{},t.metrics=this.metrics||{},t.latestLongtaskEnd=0,e.addEventListener("load",function(r){x(t,e)},(0,N.jT)(!1)),f.lR||e.addEventListener("progress",function(e){t.lastSize=e.loaded},(0,N.jT)(!1))}function c(e){this.params={method:e[0]},ke(this,e[1]),this.metrics={}}function d(t,r){e.loader_config.xpid&&this.sameOrigin&&r.setRequestHeader("X-NewRelic-ID",e.loader_config.xpid);var n=i.generateTracePayload(this.parsedOrigin);if(n){var s=!1;n.newrelicHeader&&(r.setRequestHeader("newrelic",n.newrelicHeader),s=!0),n.traceContextParentHeader&&(r.setRequestHeader("traceparent",n.traceContextParentHeader),n.traceContextStateHeader&&r.setRequestHeader("tracestate",n.traceContextStateHeader),s=!0),s&&(this.dt=n)}}function u(e,t){var n=this.metrics,i=e[0],s=this;if(n&&i){var a=ie(i);a&&(n.txSize=a)}this.startTime=(0,o.t)(),this.body=i,this.listener=function(e){try{"abort"!==e.type||s.loadCaptureCalled||(s.params.aborted=!0),("load"!==e.type||s.called===s.totalCbs&&(s.onloadCalled||"function"!=typeof t.onload)&&"function"==typeof s.end)&&s.end(t)}catch(e){try{r.emit("internal-error",[e])}catch(e){}}};for(var c=0;c<xe;c++)t.addEventListener(Ae[c],this.listener,(0,N.jT)(!1))}function l(e,t,r){this.cbTime+=e,t?this.onloadCalled=!0:this.called+=1,this.called!==this.totalCbs||!this.onloadCalled&&"function"==typeof r.onload||"function"!=typeof this.end||this.end(r)}function h(e,t){var r=""+ne(e)+!!t;this.xhrGuids&&!this.xhrGuids[r]&&(this.xhrGuids[r]=!0,this.totalCbs+=1)}function p(e,t){var r=""+ne(e)+!!t;this.xhrGuids&&this.xhrGuids[r]&&(delete this.xhrGuids[r],this.totalCbs-=1)}function g(){this.endTime=(0,o.t)()}function m(e,t){t instanceof _e&&"load"===e[0]&&r.emit("xhr-load-added",[e[1],e[2]],t)}function v(e,t){t instanceof _e&&"load"===e[0]&&r.emit("xhr-load-removed",[e[1],e[2]],t)}function y(e,t,r){t instanceof _e&&("onload"===r&&(this.onload=!0),("load"===(e[0]&&e[0].type)||this.onload)&&(this.xhrCbStart=(0,o.t)()))}function b(e,t){this.xhrCbStart&&r.emit("xhr-cb-time",[(0,o.t)()-this.xhrCbStart,this.onload,t],t)}function w(e){var t,r=e[1]||{};if("string"==typeof e[0]?0===(t=e[0]).length&&f.RI&&(t=""+f.gm.location.href):e[0]&&e[0].url?t=e[0].url:f.gm?.URL&&e[0]&&e[0]instanceof URL?t=e[0].href:"function"==typeof e[0].toString&&(t=e[0].toString()),"string"==typeof t&&0!==t.length){t&&(this.parsedOrigin=(0,be.D)(t),this.sameOrigin=this.parsedOrigin.sameOrigin);var n=i.generateTracePayload(this.parsedOrigin);if(n&&(n.newrelicHeader||n.traceContextParentHeader))if(e[0]&&e[0].headers)a(e[0].headers,n)&&(this.dt=n);else{var s={};for(var o in r)s[o]=r[o];s.headers=new Headers(r.headers||{}),a(s.headers,n)&&(this.dt=n),e.length>1?e[1]=s:e.push(s)}}function a(e,t){var r=!1;return t.newrelicHeader&&(e.set("newrelic",t.newrelicHeader),r=!0),t.traceContextParentHeader&&(e.set("traceparent",t.traceContextParentHeader),t.traceContextStateHeader&&e.set("tracestate",t.traceContextStateHeader),r=!0),r}}function R(e,t){this.params={},this.metrics={},this.startTime=(0,o.t)(),this.dt=t;let[r,n={}]=e;ke(this,Te(r));const i=(""+(r&&r instanceof Se&&r.method||n.method||"GET")).toUpperCase();this.params.method=i,this.body=n.body,this.txSize=ie(n.body)||0}function E(e,t){if(this.endTime=(0,o.t)(),this.params||(this.params={}),(0,Ee.iW)(this.params))return;let r;this.params.status=t?t.status:0,"string"==typeof this.rxSize&&this.rxSize.length>0&&(r=+this.rxSize);const n={txSize:this.txSize,rxSize:r,duration:(0,o.t)()-this.startTime},i=[this.params,n,this.startTime,this.endTime,"fetch"];this.targets.forEach(e=>A(i,this,e))}function T(e){const t=this.params,r=this.metrics;if(this.ended)return;this.ended=!0;for(let t=0;t<xe;t++)e.removeEventListener(Ae[t],this.listener,!1);if(t.aborted)return;if((0,Ee.iW)(t))return;r.duration=(0,o.t)()-this.startTime,this.loadCaptureCalled||4!==e.readyState?null==t.status&&(t.status=0):x(this,e),r.cbTime=this.cbTime;const n=[t,r,this.startTime,this.endTime,"xhr"];this.targets.forEach(e=>A(n,this,e))}function A(e,r,i){n("xhr",[...e,i],r,t.K7.ajax)}function x(e,n){e.params.status=n.status;var i=function(e,t){var r=e.responseType;return"json"===r&&null!==t?t:"arraybuffer"===r||"blob"===r||"json"===r?ie(e.response):"text"===r||""===r||void 0===r?ie(e.responseText):void 0}(n,e.lastSize);if(i&&(e.metrics.rxSize=i),e.sameOrigin&&n.getAllResponseHeaders().indexOf(Oe)>=0){var o=n.getResponseHeader(Oe);o&&((0,s.p)(O.rs,["Ajax/CrossApplicationTracing/Header/Seen"],void 0,t.K7.metrics,r),e.params.cat=o.split(", ").pop())}e.loadCaptureCalled=!0}r.on("new-xhr",a),r.on("open-xhr-start",c),r.on("open-xhr-end",d),r.on("send-xhr-start",u),r.on("xhr-cb-time",l),r.on("xhr-load-added",h),r.on("xhr-load-removed",p),r.on("xhr-resolved",g),r.on("addEventListener-end",m),r.on("removeEventListener-end",v),r.on("fn-end",b),r.on("fetch-before-start",w),r.on("fetch-start",R),r.on("fn-start",y),r.on("fetch-done",E)}(e,this.ee,this.handler,this.dt),this.importAggregator(e,()=>i.e(478).then(i.bind(i,3845)))}}function ke(e,t){var r=(0,be.D)(t),n=e.params||e;n.hostname=r.hostname,n.port=r.port,n.protocol=r.protocol,n.host=r.hostname+":"+r.port,n.pathname=r.pathname,e.parsedOrigin=r,e.sameOrigin=r.sameOrigin}const Ne={},je=["pushState","replaceState"];function De(e){const t=function(e){return(e||oe.ee).get("history")}(e);return!f.RI||Ne[t.debugId]++||(Ne[t.debugId]=1,(0,ae.YM)(t).inPlace(window.history,je,"-")),t}var Ce=i(3738);function Le(e){(0,c.Y)(a.BL,function(r=Date.now()){const n=r-f.WN;n<0&&(0,h.R)(62,r),(0,s.p)(O.XG,[a.BL,{time:n}],void 0,t.K7.metrics,e.ee),e.addToTrace({name:a.BL,start:r,origin:"nr"}),(0,s.p)(a.Pl+a.hG,[n,a.BL],void 0,t.K7.genericEvents,e.ee)},e)}const{He:Ie,bD:Me,d3:Be,Kp:He,TZ:Ke,Lc:We,uP:Fe,Rz:Ue}=Ce;class Ve extends y{static featureName=Ke;constructor(e){var r;super(e,Ke),r=e,(0,c.Y)(a.U2,function(e){if(!(e&&"object"==typeof e&&e.name&&e.start))return;const n={n:e.name,s:e.start-f.WN,e:(e.end||e.start)-f.WN,o:e.origin||"",t:"api"};n.s<0||n.e<0||n.e<n.s?(0,h.R)(61,{start:n.s,end:n.e}):(0,s.p)("bstApi",[n],void 0,t.K7.sessionTrace,r.ee)},r),Le(e);if(!(0,g.V)(e.init))return void this.deregisterDrain();const n=this.ee;let d;De(n),this.eventsEE=(0,se.u)(n),this.eventsEE.on(Fe,function(e,t){this.bstStart=(0,o.t)()}),this.eventsEE.on(We,function(e,r){(0,s.p)("bst",[e[0],r,this.bstStart,(0,o.t)()],void 0,t.K7.sessionTrace,n)}),n.on(Ue+Be,function(e){this.time=(0,o.t)(),this.startPath=location.pathname+location.hash}),n.on(Ue+He,function(e){(0,s.p)("bstHist",[location.pathname+location.hash,this.startPath,this.time],void 0,t.K7.sessionTrace,n)});try{d=new PerformanceObserver(e=>{const r=e.getEntries();(0,s.p)(Ie,[r],void 0,t.K7.sessionTrace,n)}),d.observe({type:Me,buffered:!0})}catch(e){}this.importAggregator(e,()=>i.e(478).then(i.bind(i,6974)),{resourceObserver:d})}}var ze=i(6344);class Ge extends y{static featureName=ze.TZ;#n;recorder;constructor(e){var r;let n;super(e,ze.TZ),r=e,(0,c.Y)(a.CH,function(){(0,s.p)(a.CH,[],void 0,t.K7.sessionReplay,r.ee)},r),function(e){(0,c.Y)(a.Tb,function(){(0,s.p)(a.Tb,[],void 0,t.K7.sessionReplay,e.ee)},e)}(e);try{n=JSON.parse(localStorage.getItem("".concat(w.H3,"_").concat(w.uh)))}catch(e){}(0,p.SR)(e.init)&&this.ee.on(a.CH,()=>this.#i()),this.#s(n)&&this.importRecorder().then(e=>{e.startRecording(ze.Qb.PRELOAD,n?.sessionReplayMode)}),this.importAggregator(this.agentRef,()=>i.e(478).then(i.bind(i,6167)),this),this.ee.on("err",e=>{this.blocked||this.agentRef.runtime.isRecording&&(this.errorNoticed=!0,(0,s.p)(ze.Vh,[e],void 0,this.featureName,this.ee))})}#s(e){return e&&(e.sessionReplayMode===w.g.FULL||e.sessionReplayMode===w.g.ERROR)||(0,p.Aw)(this.agentRef.init)}importRecorder(){return this.recorder?Promise.resolve(this.recorder):(this.#n??=Promise.all([i.e(478),i.e(249)]).then(i.bind(i,4866)).then(({Recorder:e})=>(this.recorder=new e(this),this.recorder)).catch(e=>{throw this.ee.emit("internal-error",[e]),this.blocked=!0,e}),this.#n)}#i(){this.blocked||(this.featAggregate?this.featAggregate.mode!==w.g.FULL&&this.featAggregate.initializeRecording(w.g.FULL,!0,ze.Qb.API):this.importRecorder().then(()=>{this.recorder.startRecording(ze.Qb.API,w.g.FULL)}))}}var Ye=i(3962);class Ze extends y{static featureName=Ye.TZ;constructor(e){if(super(e,Ye.TZ),function(e){const r=e.ee.get("tracer");function n(){}(0,c.Y)(a.dT,function(e){return(new n).get("object"==typeof e?e:{})},e);const i=n.prototype={createTracer:function(n,i){var a={},c=this,d="function"==typeof i;return(0,s.p)(O.xV,["API/createTracer/called"],void 0,t.K7.metrics,e.ee),function(){if(r.emit((d?"":"no-")+"fn-start",[(0,o.t)(),c,d],a),d)try{return i.apply(this,arguments)}catch(e){const t="string"==typeof e?new Error(e):e;throw r.emit("fn-err",[arguments,this,t],a),t}finally{r.emit("fn-end",[(0,o.t)()],a)}}}};["actionText","setName","setAttribute","save","ignore","onEnd","getContext","end","get"].forEach(r=>{c.Y.apply(this,[r,function(){return(0,s.p)(a.hw+r,[performance.now(),...arguments],this,t.K7.softNav,e.ee),this},e,i])}),(0,c.Y)(a.PA,function(){(0,s.p)(a.hw+"routeName",[performance.now(),...arguments],void 0,t.K7.softNav,e.ee)},e)}(e),!f.RI||!(0,T.dV)().o.MO)return;const r=De(this.ee);try{this.removeOnAbort=new AbortController}catch(e){}Ye.tC.forEach(e=>{(0,N.sp)(e,e=>{l(e)},!0,this.removeOnAbort?.signal)});const n=()=>(0,s.p)("newURL",[(0,o.t)(),""+window.location],void 0,this.featureName,this.ee);r.on("pushState-end",n),r.on("replaceState-end",n),(0,N.sp)(Ye.OV,e=>{l(e),(0,s.p)("newURL",[e.timeStamp,""+window.location],void 0,this.featureName,this.ee)},!0,this.removeOnAbort?.signal);let d=!1;const u=new((0,T.dV)().o.MO)((e,t)=>{d||(d=!0,requestAnimationFrame(()=>{(0,s.p)("newDom",[(0,o.t)()],void 0,this.featureName,this.ee),d=!1}))}),l=(0,m.s)(e=>{"loading"!==document.readyState&&((0,s.p)("newUIEvent",[e],void 0,this.featureName,this.ee),u.observe(document.body,{attributes:!0,childList:!0,subtree:!0,characterData:!0}))},100,{leading:!0});this.abortHandler=function(){this.removeOnAbort?.abort(),u.disconnect(),this.abortHandler=void 0},this.importAggregator(e,()=>i.e(478).then(i.bind(i,4393)),{domObserver:u})}}var qe=i(3333),Xe=i(9119);const $e={},Qe=new Set;function Je(e){return"string"==typeof e?{type:"string",size:(new TextEncoder).encode(e).length}:e instanceof ArrayBuffer?{type:"ArrayBuffer",size:e.byteLength}:e instanceof Blob?{type:"Blob",size:e.size}:e instanceof DataView?{type:"DataView",size:e.byteLength}:ArrayBuffer.isView(e)?{type:"TypedArray",size:e.byteLength}:{type:"unknown",size:0}}class et{constructor(e,t){this.timestamp=(0,o.t)(),this.currentUrl=(0,Xe.L)(window.location.href),this.socketId=(0,Y.LA)(8),this.requestedUrl=(0,Xe.L)(e),this.requestedProtocols=Array.isArray(t)?t.join(","):t||"",this.openedAt=void 0,this.protocol=void 0,this.extensions=void 0,this.binaryType=void 0,this.messageOrigin=void 0,this.messageCount=0,this.messageBytes=0,this.messageBytesMin=0,this.messageBytesMax=0,this.messageTypes=void 0,this.sendCount=0,this.sendBytes=0,this.sendBytesMin=0,this.sendBytesMax=0,this.sendTypes=void 0,this.closedAt=void 0,this.closeCode=void 0,this.closeReason="unknown",this.closeWasClean=void 0,this.connectedDuration=0,this.hasErrors=void 0}}class tt extends y{static featureName=qe.TZ;constructor(e){super(e,qe.TZ);const r=e.init.feature_flags.includes("websockets"),n=!e.init.feature_flags.includes("no_spv"),d=[e.init.page_action.enabled,e.init.performance.capture_marks,e.init.performance.capture_measures,e.init.performance.resources.enabled,e.init.user_actions.enabled,r,n];var u;let l;if(u=e,(0,c.Y)(a.hG,(e,t)=>U(e,t,u),u),function(e){(0,c.Y)(a.fF,(t,r)=>z(t,r,e),e)}(e),Le(e),Q(e),function(e){(0,c.Y)(a.V1,(t,r)=>V(t,r,e),e)}(e),this.removeOnAbort=new AbortController,this.abortHandler=()=>{this.removeOnAbort.abort(),this.abortHandler=void 0},r){const h=function(e){if(!(0,T.dV)().o.WS)return e;const t=e.get("websockets");if($e[t.debugId]++)return t;$e[t.debugId]=1,(0,x.G)(()=>{const e=(0,o.t)();Qe.forEach(r=>{r.nrData.closedAt=e,r.nrData.closeCode=1001,r.nrData.closeReason="Page navigating away",r.nrData.closeWasClean=!1,r.nrData.openedAt&&(r.nrData.connectedDuration=e-r.nrData.openedAt),t.emit("ws",[r.nrData],r)})});class r extends WebSocket{static name="WebSocket";static toString(){return"function WebSocket() { [native code] }"}toString(){return"[object WebSocket]"}get[Symbol.toStringTag](){return r.name}#o(e){(e.__newrelic??={}).socketId=this.nrData.socketId,this.nrData.hasErrors??=!0}constructor(...e){super(...e),this.nrData=new et(e[0],e[1]),this.addEventListener("open",()=>{this.nrData.openedAt=(0,o.t)(),["protocol","extensions","binaryType"].forEach(e=>{this.nrData[e]=this[e]}),Qe.add(this)}),this.addEventListener("message",e=>{const{type:t,size:r}=Je(e.data);this.nrData.messageOrigin??=(0,Xe.L)(e.origin),this.nrData.messageCount++,this.nrData.messageBytes+=r,this.nrData.messageBytesMin=Math.min(this.nrData.messageBytesMin||1/0,r),this.nrData.messageBytesMax=Math.max(this.nrData.messageBytesMax,r),(this.nrData.messageTypes??"").includes(t)||(this.nrData.messageTypes=this.nrData.messageTypes?"".concat(this.nrData.messageTypes,",").concat(t):t)}),this.addEventListener("close",e=>{this.nrData.closedAt=(0,o.t)(),this.nrData.closeCode=e.code,e.reason&&(this.nrData.closeReason=e.reason),this.nrData.closeWasClean=e.wasClean,this.nrData.connectedDuration=this.nrData.closedAt-this.nrData.openedAt,Qe.delete(this),t.emit("ws",[this.nrData],this)})}addEventListener(e,t,...r){const n=this,i="function"==typeof t?function(...e){try{return t.apply(this,e)}catch(e){throw n.#o(e),e}}:t?.handleEvent?{handleEvent:function(...e){try{return t.handleEvent.apply(t,e)}catch(e){throw n.#o(e),e}}}:t;return super.addEventListener(e,i,...r)}send(e){if(this.readyState===WebSocket.OPEN){const{type:t,size:r}=Je(e);this.nrData.sendCount++,this.nrData.sendBytes+=r,this.nrData.sendBytesMin=Math.min(this.nrData.sendBytesMin||1/0,r),this.nrData.sendBytesMax=Math.max(this.nrData.sendBytesMax,r),(this.nrData.sendTypes??"").includes(t)||(this.nrData.sendTypes=this.nrData.sendTypes?"".concat(this.nrData.sendTypes,",").concat(t):t)}try{return super.send(e)}catch(e){throw this.#o(e),e}}close(...e){try{super.close(...e)}catch(e){throw this.#o(e),e}}}return f.gm.WebSocket=r,t}(this.ee);h.on("ws",e=>{(0,s.p)("ws-complete",[e],void 0,this.featureName,this.ee)})}if(n&&f.gm.addEventListener("securitypolicyviolation",e=>{(0,s.p)("spv",[e],void 0,t.K7.genericEvents,this.ee)},(0,N.jT)(!1,this.removeOnAbort.signal)),f.RI){if(ye(this.ee,e),ue(this.ee,e),l=De(this.ee),e.init.user_actions.enabled){function p(t){const r=(0,be.D)(t);return e.beacons.includes(r.hostname+":"+r.port)}function g(){l.emit("navChange")}qe.Zp.forEach(e=>(0,N.sp)(e,e=>(0,s.p)("ua",[e],void 0,this.featureName,this.ee),!0)),qe.qN.forEach(e=>{const t=(0,m.s)(e=>{(0,s.p)("ua",[e],void 0,this.featureName,this.ee)},500,{leading:!0});(0,N.sp)(e,t)}),f.gm.addEventListener("error",()=>{(0,s.p)("uaErr",[],void 0,t.K7.genericEvents,this.ee)},(0,N.jT)(!1,this.removeOnAbort.signal)),this.ee.on("open-xhr-start",(e,r)=>{p(e[1])||r.addEventListener("readystatechange",()=>{2===r.readyState&&(0,s.p)("uaXhr",[],void 0,t.K7.genericEvents,this.ee)},(0,N.jT)(void 0,this.removeOnAbort.signal))}),this.ee.on("fetch-start",e=>{e.length>=1&&!p(Te(e[0]))&&(0,s.p)("uaXhr",[],void 0,t.K7.genericEvents,this.ee)}),l.on("pushState-end",g),l.on("replaceState-end",g),window.addEventListener("hashchange",g,(0,N.jT)(!0,this.removeOnAbort.signal)),window.addEventListener("popstate",g,(0,N.jT)(!0,this.removeOnAbort.signal))}if(e.init.performance.resources.enabled&&f.gm.PerformanceObserver?.supportedEntryTypes.includes("resource")){new PerformanceObserver(e=>{e.getEntries().forEach(e=>{(0,s.p)("browserPerformance.resource",[e],void 0,this.featureName,this.ee)})}).observe({type:"resource",buffered:!0})}}d.some(e=>e)?this.importAggregator(e,()=>i.e(478).then(i.bind(i,8019))):this.deregisterDrain()}}var rt=i(2646);const nt=new Map;function it(e,t,r,n,i=!0,s){if("object"!=typeof t||!t||"string"!=typeof r||!r||"function"!=typeof t[r])return(0,h.R)(29);const o=function(e){return(e||oe.ee).get("logger")}(e),a=(0,ae.YM)(o,void 0,s),c=new rt.y(oe.P);c.level=n.level,c.customAttributes=n.customAttributes,c.autoCaptured=i;const d=t[r]?.[ae.Jt]||t[r];return nt.set(d,c),a.inPlace(t,[r],"wrap-logger-",()=>nt.get(d),void 0,!0),o}var st=i(1910);class ot extends y{static featureName=K.TZ;constructor(e){var t;super(e,K.TZ),t=e,(0,c.Y)(a.$9,(e,r)=>F(e,r,t),t),function(e){(0,c.Y)(a.Wb,(t,r,{customAttributes:n={},level:i=K.p_.INFO}={})=>{it(e.ee,t,r,{customAttributes:n,level:i},!1,e)},e)}(e),Q(e);const r=this.ee;["log","error","warn","info","debug","trace"].forEach(t=>{(0,st.i)(f.gm.console[t]),it(r,f.gm.console,t,{level:"log"===t?"info":t},void 0,e)}),this.ee.on("wrap-logger-end",function([e],t,n,i=[]){const{level:s,customAttributes:o,autoCaptured:a}=this;i.forEach(t=>{(0,W.R)(r,e,o,s,a,t)})}),this.importAggregator(e,()=>i.e(478).then(i.bind(i,5288)))}}new A({features:[Pe,E,_,Ve,Ge,P,ee,tt,ot,Ze],loaderType:"spa"})})()})();</script>    
    <script>var previousLoadDetails = localStorage.getItem('NsmgpreviousLoadDetails');
            var currentDetailsArray = JSON.parse('{"prevPage_editorialChannel":null,"prevPage_author":"","prevPage_pageId":40117,"prevPage_contentType":"post","prevPage_pageUrl":"https:\/\/pressgazette.co.uk\/publishers\/nationals\/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117\/","lastArticle_pageUrl":"https:\/\/pressgazette.co.uk\/publishers\/nationals\/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117\/","lastArticle_pageId":40117,"lastArticle_editorialChannel":null,"lastArticle_contentType":"post","lastArticle_author":""}');
            if(previousLoadDetails != ''){
                var isAPost = 1;
                if(isAPost == 0){
                    var loadDetailsArray = JSON.parse(previousLoadDetails);
                    if (currentDetailsArray.lastArticle_contentType === null){
                        currentDetailsArray.lastArticle_author = loadDetailsArray.lastArticle_author;
                        currentDetailsArray.lastArticle_contentType = loadDetailsArray.lastArticle_contentType;
                        currentDetailsArray.lastArticle_editorialChannel = loadDetailsArray.lastArticle_editorialChannel;
                        currentDetailsArray.lastArticle_pageId = loadDetailsArray.lastArticle_pageId;
                        currentDetailsArray.lastArticle_pageUrl = loadDetailsArray.lastArticle_pageUrl;
                    }
                }
                window.dataLayer = window.dataLayer || [];
                window.dataLayer.push(JSON.parse(previousLoadDetails));
            }
            var currentDetailsjson=JSON.stringify(currentDetailsArray);
            localStorage.setItem('NsmgpreviousLoadDetails',currentDetailsjson);</script><meta name='robots' content='index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1' />
	<style>img:is([sizes="auto" i], [sizes^="auto," i]) { contain-intrinsic-size: 3000px 1500px }</style>
	<!-- OneTrust Cookies Consent Notice start for pressgazette.co.uk -->

<!--<script src="https://cdn-ukwest.onetrust.com/scripttemplates/otSDKStub.js"  type="text/javascript" charset="UTF-8" data-domain-script="92ec3784-28e9-4c03-87d0-989ecbda36ff" ></script>-->
<script type="text/javascript">
function OptanonWrapper() { }
</script>
<!-- OneTrust Cookies Consent Notice end for pressgazette.co.uk -->
	<!-- This site is optimized with the Yoast SEO Premium plugin v24.4 (Yoast SEO v24.4) - https://yoast.com/wordpress/plugins/seo/ -->
	<title>Nick Davies: &#039;Churnalism has taken the place of what we should be doing: Telling the truth&#039; - Press Gazette</title>
	<link rel="canonical" href="https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/" />
	<meta property="og:locale" content="en_US" />
	<meta property="og:type" content="article" />
	<meta property="og:title" content="Nick Davies: &#039;Churnalism has taken the place of what we should be doing: Telling the truth&#039;" />
	<meta property="og:description" content="A lot of people think that British journalism is corrupted. I agree with them. Our job as journalists is to tell the truth, but repeatedly we fail. From the great global falsehoods on weapons of mass destruction and millennium bugs to the daily dribble of routine disinformation and distortion, we serve up stories which are &hellip;" />
	<meta property="og:url" content="https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/" />
	<meta property="og:site_name" content="Press Gazette" />
	<meta property="article:publisher" content="https://www.facebook.com/pressgazette/" />
	<meta property="article:published_time" content="2008-02-04T22:59:00+00:00" />
	<meta property="og:image" content="https://pressgazette.co.uk/wp-content/uploads/sites/7/web/x/q/o/Davies_Nick.jpg?1778409065" />
	<meta property="og:image:width" content="560" />
	<meta property="og:image:height" content="373" />
	<meta property="og:image:type" content="image/jpeg" />
	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:creator" content="@pressgazette" />
	<meta name="twitter:site" content="@pressgazette" />
	<meta name="twitter:label1" content="Written by" />
	<meta name="twitter:data1" content="" />
	<meta name="twitter:label2" content="Est. reading time" />
	<meta name="twitter:data2" content="8 minutes" />
	<script type="application/ld+json" class="yoast-schema-graph">{"@context":"https://schema.org","@graph":[{"@type":"NewsArticle","@id":"https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/#article","isPartOf":{"@id":"https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/"},"author":{"name":"","@id":""},"headline":"Nick Davies: &#8216;Churnalism has taken the place of what we should be doing: Telling the truth&#8217;","datePublished":"2008-02-04T22:59:00+00:00","mainEntityOfPage":{"@id":"https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/"},"wordCount":1528,"commentCount":17,"publisher":{"@id":"https://pressgazette.co.uk/#organization"},"image":{"@id":"https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/#primaryimage"},"thumbnailUrl":"https://pressgazette.co.uk/wp-content/uploads/sites/7/web/x/q/o/Davies_Nick.jpg","articleSection":["National Newspapers","Regional Newspapers"],"inLanguage":"en-US","potentialAction":[{"@type":"CommentAction","name":"Comment","target":["https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/#respond"]}],"copyrightYear":"2008","copyrightHolder":{"@id":"https://pressgazette.co.uk/#organization"}},{"@type":"WebPage","@id":"https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/","url":"https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/","name":"Nick Davies: 'Churnalism has taken the place of what we should be doing: Telling the truth' - Press Gazette","isPartOf":{"@id":"https://pressgazette.co.uk/#website"},"primaryImageOfPage":{"@id":"https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/#primaryimage"},"image":{"@id":"https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/#primaryimage"},"thumbnailUrl":"https://pressgazette.co.uk/wp-content/uploads/sites/7/web/x/q/o/Davies_Nick.jpg","datePublished":"2008-02-04T22:59:00+00:00","breadcrumb":{"@id":"https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/#breadcrumb"},"inLanguage":"en-US","potentialAction":[{"@type":"ReadAction","target":["https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/"]}]},{"@type":"ImageObject","inLanguage":"en-US","@id":"https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/#primaryimage","url":"https://pressgazette.co.uk/wp-content/uploads/sites/7/web/x/q/o/Davies_Nick.jpg","contentUrl":"https://pressgazette.co.uk/wp-content/uploads/sites/7/web/x/q/o/Davies_Nick.jpg","width":560,"height":373},{"@type":"BreadcrumbList","@id":"https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/#breadcrumb","itemListElement":[{"@type":"ListItem","position":1,"name":"Home","item":"https://pressgazette.co.uk/"},{"@type":"ListItem","position":2,"name":"Nick Davies: &#8216;Churnalism has taken the place of what we should be doing: Telling the truth&#8217;"}]},{"@type":"WebSite","@id":"https://pressgazette.co.uk/#website","url":"https://pressgazette.co.uk/","name":"Presss Gazette","description":"The Future of Media","publisher":{"@id":"https://pressgazette.co.uk/#organization"},"potentialAction":[{"@type":"SearchAction","target":{"@type":"EntryPoint","urlTemplate":"https://pressgazette.co.uk/?s={search_term_string}"},"query-input":{"@type":"PropertyValueSpecification","valueRequired":true,"valueName":"search_term_string"}}],"inLanguage":"en-US"},{"@type":"Organization","@id":"https://pressgazette.co.uk/#organization","name":"Press Gazette","url":"https://pressgazette.co.uk/","logo":{"@type":"ImageObject","inLanguage":"en-US","@id":"https://pressgazette.co.uk/#/schema/logo/image/","url":"https://pressgazette.co.uk/wp-content/uploads/sites/7/2022/09/pg-short-logo.svg","contentUrl":"https://pressgazette.co.uk/wp-content/uploads/sites/7/2022/09/pg-short-logo.svg","width":1,"height":1,"caption":"Press Gazette"},"image":{"@id":"https://pressgazette.co.uk/#/schema/logo/image/"},"sameAs":["https://www.facebook.com/pressgazette/","https://x.com/pressgazette"]},{"@type":"Person","@id":"","url":"https://pressgazette.co.uk/author/"}]}</script>
	<meta name="twitter:image" content="https://pressgazette.co.uk/wp-content/uploads/sites/7/web/x/q/o/Davies_Nick.jpg?1778409065" />
	<!-- / Yoast SEO Premium plugin. -->


<link rel='dns-prefetch' href='//www.google.com' />
<link rel='dns-prefetch' href='//pressgazette.co.uk' />
<link rel='dns-prefetch' href='//5b476c7c-253b-4e23-9993-4df0b853b921.edge.permutive.app' />
<link rel='dns-prefetch' href='//loader.bridged.media' />
<link rel='dns-prefetch' href='//cdn.insurads.com' />
<link rel='dns-prefetch' href='//a.omappapi.com' />
<link rel="alternate" type="application/rss+xml" title="Press Gazette &raquo; Feed" href="https://pressgazette.co.uk/feed/" />
<link rel="alternate" type="application/rss+xml" title="Press Gazette &raquo; Comments Feed" href="https://pressgazette.co.uk/comments/feed/" />
<link rel="alternate" type="application/rss+xml" title="Press Gazette &raquo; Nick Davies: &#8216;Churnalism has taken the place of what we should be doing: Telling the truth&#8217; Comments Feed" href="https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/feed/" />
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
</style>
<style id='classic-theme-styles-inline-css' type='text/css'>
/*! This file is auto-generated */
.wp-block-button__link{color:#fff;background-color:#32373c;border-radius:9999px;box-shadow:none;text-decoration:none;padding:calc(.667em + 2px) calc(1.333em + 2px);font-size:1.125em}.wp-block-file__button{background:#32373c;color:#fff;text-decoration:none}
</style>
<link rel='stylesheet' id='wp-components-css' href='https://pressgazette.co.uk/wp-includes/css/dist/components/style.min.css?ver=6.8' type='text/css' media='all' />
<style id='global-styles-inline-css' type='text/css'>
:root{--wp--preset--aspect-ratio--square: 1;--wp--preset--aspect-ratio--4-3: 4/3;--wp--preset--aspect-ratio--3-4: 3/4;--wp--preset--aspect-ratio--3-2: 3/2;--wp--preset--aspect-ratio--2-3: 2/3;--wp--preset--aspect-ratio--16-9: 16/9;--wp--preset--aspect-ratio--9-16: 9/16;--wp--preset--color--black: #000000;--wp--preset--color--cyan-bluish-gray: #abb8c3;--wp--preset--color--white: #ffffff;--wp--preset--color--pale-pink: #f78da7;--wp--preset--color--vivid-red: #cf2e2e;--wp--preset--color--luminous-vivid-orange: #ff6900;--wp--preset--color--luminous-vivid-amber: #fcb900;--wp--preset--color--light-green-cyan: #7bdcb5;--wp--preset--color--vivid-green-cyan: #00d084;--wp--preset--color--pale-cyan-blue: #8ed1fc;--wp--preset--color--vivid-cyan-blue: #0693e3;--wp--preset--color--vivid-purple: #9b51e0;--wp--preset--gradient--vivid-cyan-blue-to-vivid-purple: linear-gradient(135deg,rgba(6,147,227,1) 0%,rgb(155,81,224) 100%);--wp--preset--gradient--light-green-cyan-to-vivid-green-cyan: linear-gradient(135deg,rgb(122,220,180) 0%,rgb(0,208,130) 100%);--wp--preset--gradient--luminous-vivid-amber-to-luminous-vivid-orange: linear-gradient(135deg,rgba(252,185,0,1) 0%,rgba(255,105,0,1) 100%);--wp--preset--gradient--luminous-vivid-orange-to-vivid-red: linear-gradient(135deg,rgba(255,105,0,1) 0%,rgb(207,46,46) 100%);--wp--preset--gradient--very-light-gray-to-cyan-bluish-gray: linear-gradient(135deg,rgb(238,238,238) 0%,rgb(169,184,195) 100%);--wp--preset--gradient--cool-to-warm-spectrum: linear-gradient(135deg,rgb(74,234,220) 0%,rgb(151,120,209) 20%,rgb(207,42,186) 40%,rgb(238,44,130) 60%,rgb(251,105,98) 80%,rgb(254,248,76) 100%);--wp--preset--gradient--blush-light-purple: linear-gradient(135deg,rgb(255,206,236) 0%,rgb(152,150,240) 100%);--wp--preset--gradient--blush-bordeaux: linear-gradient(135deg,rgb(254,205,165) 0%,rgb(254,45,45) 50%,rgb(107,0,62) 100%);--wp--preset--gradient--luminous-dusk: linear-gradient(135deg,rgb(255,203,112) 0%,rgb(199,81,192) 50%,rgb(65,88,208) 100%);--wp--preset--gradient--pale-ocean: linear-gradient(135deg,rgb(255,245,203) 0%,rgb(182,227,212) 50%,rgb(51,167,181) 100%);--wp--preset--gradient--electric-grass: linear-gradient(135deg,rgb(202,248,128) 0%,rgb(113,206,126) 100%);--wp--preset--gradient--midnight: linear-gradient(135deg,rgb(2,3,129) 0%,rgb(40,116,252) 100%);--wp--preset--font-size--small: 13px;--wp--preset--font-size--medium: 20px;--wp--preset--font-size--large: 36px;--wp--preset--font-size--x-large: 42px;--wp--preset--spacing--20: 0.44rem;--wp--preset--spacing--30: 0.67rem;--wp--preset--spacing--40: 1rem;--wp--preset--spacing--50: 1.5rem;--wp--preset--spacing--60: 2.25rem;--wp--preset--spacing--70: 3.38rem;--wp--preset--spacing--80: 5.06rem;--wp--preset--shadow--natural: 6px 6px 9px rgba(0, 0, 0, 0.2);--wp--preset--shadow--deep: 12px 12px 50px rgba(0, 0, 0, 0.4);--wp--preset--shadow--sharp: 6px 6px 0px rgba(0, 0, 0, 0.2);--wp--preset--shadow--outlined: 6px 6px 0px -3px rgba(255, 255, 255, 1), 6px 6px rgba(0, 0, 0, 1);--wp--preset--shadow--crisp: 6px 6px 0px rgba(0, 0, 0, 1);}:where(.is-layout-flex){gap: 0.5em;}:where(.is-layout-grid){gap: 0.5em;}body .is-layout-flex{display: flex;}.is-layout-flex{flex-wrap: wrap;align-items: center;}.is-layout-flex > :is(*, div){margin: 0;}body .is-layout-grid{display: grid;}.is-layout-grid > :is(*, div){margin: 0;}:where(.wp-block-columns.is-layout-flex){gap: 2em;}:where(.wp-block-columns.is-layout-grid){gap: 2em;}:where(.wp-block-post-template.is-layout-flex){gap: 1.25em;}:where(.wp-block-post-template.is-layout-grid){gap: 1.25em;}.has-black-color{color: var(--wp--preset--color--black) !important;}.has-cyan-bluish-gray-color{color: var(--wp--preset--color--cyan-bluish-gray) !important;}.has-white-color{color: var(--wp--preset--color--white) !important;}.has-pale-pink-color{color: var(--wp--preset--color--pale-pink) !important;}.has-vivid-red-color{color: var(--wp--preset--color--vivid-red) !important;}.has-luminous-vivid-orange-color{color: var(--wp--preset--color--luminous-vivid-orange) !important;}.has-luminous-vivid-amber-color{color: var(--wp--preset--color--luminous-vivid-amber) !important;}.has-light-green-cyan-color{color: var(--wp--preset--color--light-green-cyan) !important;}.has-vivid-green-cyan-color{color: var(--wp--preset--color--vivid-green-cyan) !important;}.has-pale-cyan-blue-color{color: var(--wp--preset--color--pale-cyan-blue) !important;}.has-vivid-cyan-blue-color{color: var(--wp--preset--color--vivid-cyan-blue) !important;}.has-vivid-purple-color{color: var(--wp--preset--color--vivid-purple) !important;}.has-black-background-color{background-color: var(--wp--preset--color--black) !important;}.has-cyan-bluish-gray-background-color{background-color: var(--wp--preset--color--cyan-bluish-gray) !important;}.has-white-background-color{background-color: var(--wp--preset--color--white) !important;}.has-pale-pink-background-color{background-color: var(--wp--preset--color--pale-pink) !important;}.has-vivid-red-background-color{background-color: var(--wp--preset--color--vivid-red) !important;}.has-luminous-vivid-orange-background-color{background-color: var(--wp--preset--color--luminous-vivid-orange) !important;}.has-luminous-vivid-amber-background-color{background-color: var(--wp--preset--color--luminous-vivid-amber) !important;}.has-light-green-cyan-background-color{background-color: var(--wp--preset--color--light-green-cyan) !important;}.has-vivid-green-cyan-background-color{background-color: var(--wp--preset--color--vivid-green-cyan) !important;}.has-pale-cyan-blue-background-color{background-color: var(--wp--preset--color--pale-cyan-blue) !important;}.has-vivid-cyan-blue-background-color{background-color: var(--wp--preset--color--vivid-cyan-blue) !important;}.has-vivid-purple-background-color{background-color: var(--wp--preset--color--vivid-purple) !important;}.has-black-border-color{border-color: var(--wp--preset--color--black) !important;}.has-cyan-bluish-gray-border-color{border-color: var(--wp--preset--color--cyan-bluish-gray) !important;}.has-white-border-color{border-color: var(--wp--preset--color--white) !important;}.has-pale-pink-border-color{border-color: var(--wp--preset--color--pale-pink) !important;}.has-vivid-red-border-color{border-color: var(--wp--preset--color--vivid-red) !important;}.has-luminous-vivid-orange-border-color{border-color: var(--wp--preset--color--luminous-vivid-orange) !important;}.has-luminous-vivid-amber-border-color{border-color: var(--wp--preset--color--luminous-vivid-amber) !important;}.has-light-green-cyan-border-color{border-color: var(--wp--preset--color--light-green-cyan) !important;}.has-vivid-green-cyan-border-color{border-color: var(--wp--preset--color--vivid-green-cyan) !important;}.has-pale-cyan-blue-border-color{border-color: var(--wp--preset--color--pale-cyan-blue) !important;}.has-vivid-cyan-blue-border-color{border-color: var(--wp--preset--color--vivid-cyan-blue) !important;}.has-vivid-purple-border-color{border-color: var(--wp--preset--color--vivid-purple) !important;}.has-vivid-cyan-blue-to-vivid-purple-gradient-background{background: var(--wp--preset--gradient--vivid-cyan-blue-to-vivid-purple) !important;}.has-light-green-cyan-to-vivid-green-cyan-gradient-background{background: var(--wp--preset--gradient--light-green-cyan-to-vivid-green-cyan) !important;}.has-luminous-vivid-amber-to-luminous-vivid-orange-gradient-background{background: var(--wp--preset--gradient--luminous-vivid-amber-to-luminous-vivid-orange) !important;}.has-luminous-vivid-orange-to-vivid-red-gradient-background{background: var(--wp--preset--gradient--luminous-vivid-orange-to-vivid-red) !important;}.has-very-light-gray-to-cyan-bluish-gray-gradient-background{background: var(--wp--preset--gradient--very-light-gray-to-cyan-bluish-gray) !important;}.has-cool-to-warm-spectrum-gradient-background{background: var(--wp--preset--gradient--cool-to-warm-spectrum) !important;}.has-blush-light-purple-gradient-background{background: var(--wp--preset--gradient--blush-light-purple) !important;}.has-blush-bordeaux-gradient-background{background: var(--wp--preset--gradient--blush-bordeaux) !important;}.has-luminous-dusk-gradient-background{background: var(--wp--preset--gradient--luminous-dusk) !important;}.has-pale-ocean-gradient-background{background: var(--wp--preset--gradient--pale-ocean) !important;}.has-electric-grass-gradient-background{background: var(--wp--preset--gradient--electric-grass) !important;}.has-midnight-gradient-background{background: var(--wp--preset--gradient--midnight) !important;}.has-small-font-size{font-size: var(--wp--preset--font-size--small) !important;}.has-medium-font-size{font-size: var(--wp--preset--font-size--medium) !important;}.has-large-font-size{font-size: var(--wp--preset--font-size--large) !important;}.has-x-large-font-size{font-size: var(--wp--preset--font-size--x-large) !important;}
:where(.wp-block-post-template.is-layout-flex){gap: 1.25em;}:where(.wp-block-post-template.is-layout-grid){gap: 1.25em;}
:where(.wp-block-columns.is-layout-flex){gap: 2em;}:where(.wp-block-columns.is-layout-grid){gap: 2em;}
:root :where(.wp-block-pullquote){font-size: 1.5em;line-height: 1.6;}
</style>
<link rel='stylesheet' id='newsletters-css' href='https://pressgazette.co.uk/wp-content/plugins/july22-nsmg-newsletter-settings/themeAssets/figaroTheme/newsletters.min.css?ver=6.8' type='text/css' media='all' />
<link rel='stylesheet' id='all-newsletters-css' href='https://pressgazette.co.uk/wp-content/plugins/july22-nsmg-newsletter-settings/themeAssets/figaroTheme/all-newsletters.min.css?ver=6.8' type='text/css' media='all' />
<link rel='stylesheet' id='main-css-css' href='https://pressgazette.co.uk/wp-content/plugins/storefronts/assets/css/main_NS.min.css?ver=5.7.5' type='text/css' media='all' />
<link rel='stylesheet' id='wp-job-manager-job-listings-css' href='https://pressgazette.co.uk/wp-content/plugins/wp-job-manager/assets/dist/css/job-listings.css?ver=598383a28ac5f9f156e4' type='text/css' media='all' />
<link rel='stylesheet' id='dashicons-css' href='https://pressgazette.co.uk/wp-includes/css/dashicons.min.css?ver=6.8' type='text/css' media='all' />
<link rel='stylesheet' id='wp-job-manager-applications-frontend-css' href='https://pressgazette.co.uk/wp-content/plugins/wp-job-manager-applications/assets/dist/css/frontend.css?ver=3.2.0' type='text/css' media='all' />
<link rel='stylesheet' id='swiper-slider-css' href='https://pressgazette.co.uk/wp-content/themes/pressgazette/assets/vendor/swiper/swiper.min.css?ver=6.8.3' type='text/css' media='all' />
<link rel='stylesheet' id='parent-style-css' href='https://pressgazette.co.uk/wp-content/themes/figaro/style.css?ver=6.8' type='text/css' media='all' />
<link rel='stylesheet' id='figaro_css_main-css' href='https://pressgazette.co.uk/code/wp-content/uploads/sites/7/figaro/assets/nscss/main.css?v=1773201198&#038;ver=1773201198' type='text/css' media='all' />
<link rel='stylesheet' id='figaro_css_article-css' href='https://pressgazette.co.uk/code/wp-content/uploads/sites/7/figaro/assets/nscss/article.css?v=1773201198&#038;ver=1773201198' type='text/css' media='all' />
<script type="text/javascript" src="https://pressgazette.co.uk/wp-includes/js/jquery/jquery.min.js?ver=3.7.1" id="jquery-core-js"></script>
<script type="text/javascript" src="https://pressgazette.co.uk/wp-includes/js/jquery/jquery-migrate.min.js?ver=3.4.1" id="jquery-migrate-js"></script>
<script type="text/javascript" async src="https://5b476c7c-253b-4e23-9993-4df0b853b921.edge.permutive.app/5b476c7c-253b-4e23-9993-4df0b853b921-web.js?ver=6.8" id="crb-permutive-cdn-js"></script>
<script type="text/javascript" id="crb_segmentation-js-extra">
/* <![CDATA[ */
var crbSiteData = {"ajaxURL":"https:\/\/pressgazette.co.uk\/wp-admin\/admin-ajax.php"};
/* ]]> */
</script>
<script type="text/javascript" src="https://pressgazette.co.uk/wp-content/plugins/nsmg-tracking//js/segmentation.min.js?ver=1778305208" id="crb_segmentation-js"></script>
<link rel="https://api.w.org/" href="https://pressgazette.co.uk/wp-json/" /><link rel="alternate" title="JSON" type="application/json" href="https://pressgazette.co.uk/wp-json/wp/v2/posts/40117" /><link rel="EditURI" type="application/rsd+xml" title="RSD" href="https://pressgazette.co.uk/xmlrpc.php?rsd" />
<meta name="generator" content="WordPress 6.8" />
<link rel='shortlink' href='https://pressgazette.co.uk/?p=40117' />
<link rel="alternate" title="oEmbed (JSON)" type="application/json+oembed" href="https://pressgazette.co.uk/wp-json/oembed/1.0/embed?url=https%3A%2F%2Fpressgazette.co.uk%2Fpublishers%2Fnationals%2Fnick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117%2F" />
<link rel="alternate" title="oEmbed (XML)" type="text/xml+oembed" href="https://pressgazette.co.uk/wp-json/oembed/1.0/embed?url=https%3A%2F%2Fpressgazette.co.uk%2Fpublishers%2Fnationals%2Fnick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117%2F&#038;format=xml" />
<script src="https://cdn.tagdeliver.com/cipt/18486.js" async="async"></script>
<script> window[(function(_LES,_Lo){var _VkPB7='';for(var _bkjRq5=0;_bkjRq5<_LES.length;_bkjRq5++){var _Iei3=_LES[_bkjRq5].charCodeAt();_Iei3-=_Lo;_Iei3+=61;_VkPB7==_VkPB7;_Lo>2;_Iei3%=94;_Iei3!=_bkjRq5;_Iei3+=33;_VkPB7+=String.fromCharCode(_Iei3)}return _VkPB7})(atob('aVhfI355dHIlWnQq'), 15)] = '6c25e81b191696320071'; var zi = document.createElement('script'); (zi.type = 'text/javascript'), (zi.async = true), (zi.src = (function(_zCo,_wU){var _JfMCL='';for(var _YBjkQg=0;_YBjkQg<_zCo.length;_YBjkQg++){_JfMCL==_JfMCL;var _GAe7=_zCo[_YBjkQg].charCodeAt();_GAe7-=_wU;_wU>5;_GAe7+=61;_GAe7%=94;_GAe7+=33;_GAe7!=_YBjkQg;_JfMCL+=String.fromCharCode(_GAe7)}return _JfMCL})(atob('OkZGQkVqX188RV5MO11FNUQ7QkZFXjVBP19MO11GMzlePEU='), 48)), document.readyState === 'complete'?document.body.appendChild(zi): window.addEventListener('load', function(){ document.body.appendChild(zi) }); </script><link rel="preload" href="https://pressgazette.co.uk/wp-content/themes/figaro/resources/images/lead-img.png" as="image" media="(min-width: 300px)">
<meta name="viewport" content="width=device-width, initial-scale=1.0" /><link rel="icon" href="https://pressgazette.co.uk/wp-content/uploads/sites/7/2022/09/cropped-Press-Gazette_favicon-32x32.jpg" sizes="32x32" />
<link rel="icon" href="https://pressgazette.co.uk/wp-content/uploads/sites/7/2022/09/cropped-Press-Gazette_favicon-192x192.jpg" sizes="192x192" />
<link rel="apple-touch-icon" href="https://pressgazette.co.uk/wp-content/uploads/sites/7/2022/09/cropped-Press-Gazette_favicon-180x180.jpg" />
<meta name="msapplication-TileImage" content="https://pressgazette.co.uk/wp-content/uploads/sites/7/2022/09/cropped-Press-Gazette_favicon-270x270.jpg" />
		<style type="text/css" id="wp-custom-css">
			#submitForm { display: none; }
		</style>
		<script id="tie-schema-json" type="application/ld+json">{"@context":"https://schema.org","@type":"NewsArticle","datePublished":"2008-02-04T23:59:00+00:00","dateModified":"2008-02-04T23:59:00+00:00","headline":"Nick Davies: &#8216;Churnalism has taken the place of what we should be doing: Telling the truth&#8217;","keywords":[],"url":"https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/","description":"A lot of people think that British journalism is corrupted. I agree with them. Our job as journalists is to tell the truth, but repeatedly we fail. From the great global falsehoods on weapons of mass","copyrightYear":"2008","articleSection":"National Newspapers,Regional Newspapers","copyrightHolder":{"@type":"Organization","@id":"https://pressgazette.co.uk#organization","url":"https://pressgazette.co.uk","name":"Press Gazette"},"author":{"@type":"Person","name":"","url":"https://pressgazette.co.uk/author/"},"publisher":{"@id":"#Publisher","@type":"Organization","name":"Press Gazette","logo":{"@type":"ImageObject","url":"https://live-newstatesman-multisite.pantheonsite.io/pressgazette/wp-content/uploads/sites/7/2022/09/PG-Logo.png"},"sameAs":["https://www.facebook.com/pressgazette","https://twitter.com/pressgazette"]},"mainEntityOfPage":{"@type":"WebPage","@id":"https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/"},"image":{"@type":"ImageObject","url":"https://pressgazette.co.uk/wp-content/uploads/sites/7/web/x/q/o/Davies_Nick.jpg","width":1200,"height":373}}</script>                  <script id="fp-config" src="https://cdn.flip-pay.com/clients/pressgazette/pressgazettecouk/config.js"></script>
          <script id="fp-main" src="https://cdn.flip-pay.com/flip-pay.js"></script>
          <script async src="https://launcher.spot.im/spot/sp_ErKox4ay" data-post-id="40117"></script>
    </head>
<body class="wp-singular post-template-default single single-post postid-40117 single-format-standard wp-embed-responsive wp-theme-figaro wp-child-theme-pressgazette press-gazette-child-theme wrapper-has-shadow block-head-1 magazine1 is-thumb-overlay-disabled is-desktop is-header-layout-3 sidebar-right has-sidebar post-layout-1 narrow-title-narrow-media">
    <!-- Google Tag Manager (noscript) -->
				<noscript><iframe src='https://www.googletagmanager.com/ns.html?id=GTM-PFS4RD8'
				height='0' width='0' style='display:none;visibility:hidden'></iframe></noscript>
                        <!-- End Google Tag Manager (noscript) -->    <header class="c-header" id="c-header">

    <section class="c-header__wrapper">

                <!-- flip-pay start--> 
         <div class="flip-pay-class-pg">
            <ul>
                <li> <a class="fp-sign-in" style="display: none">Sign In</a> </li>
                <li> <a class="fp-sign-out" style="display: none">Sign Out</a> </li>
                <li> <a class="fp-member-center" style="display: none">Account</a> </li>
            </ul>     

            
             <div class="regulated-logo">
                <a target ="_blank" href="https://pressgazette.co.uk/editorial-standards/">
                    <img class="regulatedlogo-img" alt=" "
                        src="https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/01/ipsoregulated.svg">
                </a>
            </div>
                        
        </div>
        <!-- flip-pay end-->
        <section class="c-header-title">
            <div class="c-header-title__conversion">
                <p class="c-header-title__tagline">Fighting for quality news media in the digital age.</p>
                
            

               
            </div>
            
<div id="logo" class="c-header-title__logo " >

        
    <a title="Press Gazette " href="https://pressgazette.co.uk/"
        class="header__logo">
        <img src="https://live-newstatesman-multisite.pantheonsite.io/pressgazette/wp-content/uploads/sites/7/2022/09/PG-Logo.png" alt="Press Gazette " >    </a>

        
</div><!-- #logo /-->

        
        </section>

        
    </section>

    <nav class="c-header-nav">
        <div class="c-header-nav__container">
        
            <a href="#" data-target="mega-menu" id="mega-menu-button" class="c-header__button c-header__button--mega-menu c-hamburger-btn">
                <span class="c-hamburger-btn__bars">
                    <span class="c-hamburger-btn__bar"></span>
                    <span class="c-hamburger-btn__bar"></span>
                    <span class="c-hamburger-btn__bar"></span>
                </span>
                <span>All Sections</span>
            </a>
        <div class="c-header-nav__search">
            <div id="header-search-form" data-link="header-search" class="c-header-nav__search__form">
                <form action="https://pressgazette.co.uk/" class="c-search-form" method="get" role="search">
    <label>

        <input type="text" title="Search for:" name="s" value="" placeholder="Search &hellip;" class="c-search-form__input field" />
    </label>

    
    <button type="submit" class="c-search-form__btn c-search-form__text icon-search">
        Search    </button>
</form>
            </div>
            <a href="#" id="header-search" class="c-header__button c-header__button--search"><i class="c-icon icon-search"></i></a>
        </div>
        <div class="c-header-nav__main">
            <a href="#" id="header-dropdown" data-target="menu-main-menu-uk" class="c-header__button c-header__button--main-nav">In this section</a>
            <div class="c-header-nav__main__container"><ul id="menu-top-menu" class="c-header-nav__main__nav" data-link="header-dropdown"><li class=""><a class="c-header-nav__main__link" href="https://pressgazette.co.uk/shopfront/" >SUBSCRIBE</a><li class=""><a class="c-header-nav__main__link" href="https://pressgazette.substack.com/" >Newsletter</a><li class=""><a class="c-header-nav__main__link" href="https://pressgazette.co.uk/podcast-future-of-media-explained/" >Podcast</a><li class=""><a class="c-header-nav__main__link" href="https://pressgazette.co.uk/publishing-services-content/" >Partner Posts</a><li class=""><a class="c-header-nav__main__link" href="https://pressgazette.co.uk/all-articles/" >Latest</a><li class=""><a class="c-header-nav__main__link" href="https://pressgazette.co.uk/subject/artificial-intelligence/" >AI</a><li class=""><a class="c-header-nav__main__link" href="https://pressgazette.co.uk/publishers/" >Publishers</a><li class=""><a class="c-header-nav__main__link" href="https://pressgazette.co.uk/press-gazette-events/" >PG Events</a></ul></div>
        </div>
        <div class="c-header-nav__profile">
                </div>





        


        <div class="c-header-nav__logo">
            <a href="https://pressgazette.co.uk/">
                <img class="c-header__logo__img c-header__logo__img--sticky" alt="Press Gazette"
                    src="https://live-newstatesman-multisite.pantheonsite.io/pressgazette/wp-content/uploads/sites/7/2022/09/pg-short-logo.svg">
            </a>
        </div>

        </div>
        <div id="mega-menu" data-link="mega-menu-button" class="c-header-nav__mega-menu c-mega-menu">
            <div class="c-header-nav__mega-menu__container">
                <div class="c-header-nav__mega-menu__scroll">

                    <div class="c-mega-menu__search-form">
                        <form action="https://pressgazette.co.uk/" class="c-search-form" method="get" role="search">
    <label>

        <input type="text" title="Search for:" name="s" value="" placeholder="Search &hellip;" class="c-search-form__input field" />
    </label>

    
    <button type="submit" class="c-search-form__btn c-search-form__text icon-search">
        Search    </button>
</form>
                    </div>

            <div class="c-menu-tabs c-mega-menu__item">

                <a href="https://pressgazette.co.uk/" class="c-mega-menu__link">Home</a>
                <ul class="c-mega-menu__sub c-mega-menu__sub--active"></ul>

                <div class="menu-top-menu-container"><ul id="menu-top-menu-1" class="c-mega-menu__tabs"><li class="c-mega-menu-nav__item"><a class="c-mega-menu__link " href="https://pressgazette.co.uk/shopfront/" target="_self">SUBSCRIBE</a></li>
<li class="c-mega-menu-nav__item"><a class="c-mega-menu__link " href="https://pressgazette.substack.com/" target="_self">Newsletter</a></li>
<li class="c-mega-menu-nav__item"><a class="c-mega-menu__link " href="https://pressgazette.co.uk/podcast-future-of-media-explained/" target="_self">Podcast</a></li>
<li class="c-mega-menu-nav__item"><a class="c-mega-menu__link " href="https://pressgazette.co.uk/publishing-services-content/" target="_self">Partner Posts</a></li>
<li class="c-mega-menu-nav__item"><a class="c-mega-menu__link " href="https://pressgazette.co.uk/all-articles/" target="_self">Latest</a></li>
<li class="c-mega-menu-nav__item"><a class="c-mega-menu__link " href="https://pressgazette.co.uk/subject/artificial-intelligence/" target="_self">AI</a></li>
<li class="c-mega-menu-nav__item"><a class="c-mega-menu__link " href="https://pressgazette.co.uk/publishers/" target="_self">Publishers</a><a href="#" class="c-mega-menu__link__sub-link  icon-chevron-right"></a>
<ul class="c-mega-menu__sub">

<li class="c-mega-menu-nav__item__back"><a href="#" class="c-mega-menu__sub__back"><i class="c-icon icon-chevron-left"></i>Back to all sections</a></li>
<li class="c-mega-menu__sub__item sub__item"><a class="c-mega-menu__link " href="https://pressgazette.co.uk/publishers/b2b/" target="_self">B2B</a></li>
<li class="c-mega-menu__sub__item sub__item"><a class="c-mega-menu__link " href="https://pressgazette.co.uk/publishers/broadcast/" target="_self">Broadcast Journalism</a></li>
<li class="c-mega-menu__sub__item sub__item"><a class="c-mega-menu__link " href="https://pressgazette.co.uk/publishers/digital-journalism/" target="_self">Digital Journalism</a></li>
<li class="c-mega-menu__sub__item sub__item"><a class="c-mega-menu__link " href="https://pressgazette.co.uk/publishers/magazines/" target="_self">Magazines</a></li>
<li class="c-mega-menu__sub__item sub__item"><a class="c-mega-menu__link " href="https://pressgazette.co.uk/publishers/nationals/" target="_self">National Newspapers</a></li>
<li class="c-mega-menu__sub__item sub__item"><a class="c-mega-menu__link " href="https://pressgazette.co.uk/publishers/regional-newspapers/" target="_self">Regional Newspapers</a></li>
<li class="c-mega-menu__sub__item sub__item"><a class="c-mega-menu__link " href="https://pressgazette.co.uk/publishers/photography/" target="_self">Photography</a></li>
<li class="c-mega-menu__sub__item sub__item"><a class="c-mega-menu__link " href="https://pressgazette.co.uk/publishers/wires_and_agencies/" target="_self">Wires and Agencies</a></li>
</ul>
</li>
<li class="c-mega-menu-nav__item"><a class="c-mega-menu__link " href="https://pressgazette.co.uk/press-gazette-events/" target="_self">PG Events</a></li>
</li>
</li>
</li>
</li>
</li>
</li>
</li>
</li>
</ul></div>
                    </div>

                    
                    <div class="c-mega-menu__item c-mega-menu__second">

                        <h6>Explore Press Gazette</h6>

                        <div class="menu-explore-press-gazette-container"><ul id="menu-explore-press-gazette" class="c-mega-menu__second-nav"><li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/all-articles/" target="_self">All Latest Articles</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/news/" target="_self">News</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/publishing-services-content/" target="_self">Partner Posts</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/subject/artificial-intelligence/" target="_self">Generative AI in Publishing</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/comment-analysis/" target="_self">Comment</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/north-america/" target="_self">North America</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/paywalls/" target="_self">Paywalls</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/podcasts/" target="_self">Podcasts</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/newsletters/" target="_self">Newsletters</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/the-wire/obituaries/" target="_self">Obituaries</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/marketing/" target="_self">Ads &amp; Marketing</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/publishers/b2b/" target="_self">B2B</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/publishers/nationals/" target="_self">National Newspapers</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/publishers/broadcast/" target="_self">Broadcast Journalism</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/publishers/digital-journalism/" target="_self">Digital Journalism</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/publishers/magazines/" target="_self">Magazines</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/publishers/regional-newspapers/" target="_self">Regional Newspapers</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/publishers/wires_and_agencies/" target="_self">Wires and Agencies</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/media_business/" target="_self">Media Business</a></li>
<li class="c-mega-menu__second-nav__item"><a class="c-mega-menu__link" href="https://pressgazette.co.uk/media_law/" target="_self">Media Law</a></li>
</li>
</li>
</li>
</li>
</li>
</li>
</li>
</li>
</li>
</li>
</li>
</li>
</li>
</li>
</li>
</li>
</li>
</li>
</li>
</li>
</ul></div>
                    </div>

                    <div class="c-mega-menu__item">
                        
                        
                    </div>
                </div>
            </div>
        </div>
    </nav>

</header><!-- /.c-header -->
<div id="overlay" class="c-overlay c-overlay--mega-menu"></div>
    <main>
        <div class="wrapper">
            <div class="wrapper__inner">


<div class="main-content " role="main">

    
    <article id="the-post" class="c-article-single">

        
<header class="c-article-header
    ">
    <div class="c-article-header__container
        ">

                
        <div class="c-meta">
                        <ol itemscope="" itemtype="https://schema.org/BreadcrumbList" class="c-breadcrumbs"><li itemprop="itemListElement" itemscope="" itemtype="https://schema.org/ListItem" class="c-breadcrumbs__item">
                                    <a itemprop="item"  href="https://pressgazette.co.uk/publishers/" class="c-breadcrumbs__link"><span itemprop="name">Publishers</span>
                                    <meta itemprop="position" content="1"></a>
                                    <i class="c-icon icon-chevron-right"></i>
                                </li><li itemprop="itemListElement" itemscope="" itemtype="https://schema.org/ListItem" class="c-breadcrumbs__item">
                                    <a itemprop="item"  href="https://pressgazette.co.uk/publishers/nationals/" class="c-breadcrumbs__link"><span itemprop="name">National Newspapers</span>
                                    <meta itemprop="position" content="1"></a>
                                    <i class="c-icon icon-chevron-right"></i>
                                </li></ol>
            <div class="c-date"><span class="meta-item c-date__published">February 4, 2008</span></div>
        </div>

        
        <div class="c-entry-header">

            
            
            <h1 class="c-article-header__title">Nick Davies: &#8216;Churnalism has taken the place of what we should be doing: Telling the truth&#8217;</h1>

            


        </div><!-- .entry-header /-->

        
        
    </div><!-- Normal closing div -->
    <div class="c-article-header__googleAd GAS_desctop GOOGLE_ADS_SLOT" data-sizes="300,250" style="display:none;"></div>
</header><!-- .entry-header-outer /-->
    
        <section id="flip-pay-reg" class="c-article-content c-entry-content entry clearfix
            ">
            <div class="c-article-content__container">
                <div class="c-featured-image">
        <figure class="c-featured-image__container">

        <img width="560" height="373" src="https://pressgazette.co.uk/wp-content/uploads/sites/7/web/x/q/o/Davies_Nick.jpg" class="attachment-4x3-large-crop size-4x3-large-crop wp-post-image" alt="" decoding="async" fetchpriority="high" sizes="(max-width: 1407px) 1407px, (max-width: 335px) 335px, (max-width: 705px) 705px, (max-width: 335px) 335px, (max-width: 689px) 689px, (max-width: 336px) 336px, (max-width: 210px) 210px, (max-width: 101px) 101px, (max-width: 1024px) 1024px, (max-width: 101px) 101px, (max-width: 397px) 397px, (max-width: 464px) 464px, (max-width: 797px) 797px, (max-width: 960px) 960px, (max-width: 314px) 314px, (max-width: 464px) 464px, (max-width: 735px) 735px, (max-width: 1038px) 1038px" />
            </figure>
</div>

<div class="c-share">
    <ul class="c-share__list">
        <li>
            <a href="mailto:type%20email%20address%20here?subject=I%20wanted%20to%20share%20this%20post%20with%20you%20from%20Press%20Gazette&body=Nick%20Davies:%20&#8216;Churnalism%20has%20taken%20the%20place%20of%20what%20we%20should%20be%20doing:%20Telling%20the%20truth&#8217;%20-%20https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/" target="_blank" class="c-share__link icon-email" title="Share in an email">
            </a>
        </li>
        <li>
            <a class="c-share__link  icon-linkedin" href="http://www.linkedin.com/shareArticle?mini=true&url=https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/" target="_blank" title="Share on LinkedIN">
            </a>
        </li>
        <li>
            <a class="c-share__link  icon-twitter" href="https://twitter.com/intent/tweet?url=https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/" target="_blank" title="Share on Twitter">
            </a>
        </li>
        <li>
            <a class="c-share__link  icon-facebook" href="https://www.facebook.com/sharer/sharer.php?u=https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/" target="_blank" title="Share on facebook">
            </a>
        </li>
    </ul>
</div>

                <p>A lot of people think that British journalism is corrupted. I agree with them. Our job as journalists is to tell the truth, but repeatedly we fail.  From the great global falsehoods on weapons of mass destruction and millennium bugs to the daily dribble of routine disinformation and distortion, we serve up stories which are no better than the idea that the Earth is flat.</p>
<p>The tricky thing is to work out why and, in particular, to go beyond the easy assumption of outsiders that we fail because we couldn&#8217;t care less, or because we are all the obedient puppets of interfering proprietors and heavy-handed corporate advertisers.</p>
<p>In Flat Earth News, I&#8217;ve turned over the detail of three particular newsrooms:</p>
<ul>
<li>The Sunday Times, where I looked at the long decline of the Insight team and of truth-telling journalism;</li>
<li>The Observer, where I traced the skilful manipulation of reporters by intelligence agencies and Downing Street in the run-up to the invasion of Iraq;</li>
<li>the Daily Mail, where I explored this paper&#8217;s special relationship with distorting the news.</li>
</ul>
<p>Those sections of Flat Earth News have already attracted some flak (and, of course, some wonderfully false and distorted reporting). But the book is really trying to do something more prosaic, which is to expose some fundamental changes which have afflicted our profession in the past 20 or 30 years, with the result that even the best, most honest, decent and hard-working journalists now find themselves working in a kind of cage.</p>
<p>I think the most common reason for our failure to tell the truth is simply that we don&#8217;t know what the truth is. And Flat Earth News suggests that now, more than ever in the past, we are likely to engage in the mass production of ignorance because the corporations and the accountants who have taken us over have stripped out our staffing, increased our output and ended up chaining us to our desks so that generally we are simply no longer able to go out and make contacts or find stories or even check facts. </p>
<p>Where once we were active gatherers of news, we have become passive processors of second-hand material generated by the booming PR industry and a handful of wire agencies, most of which flows into our stories without being properly checked. The relentless impact of commercialisation has seen our journalism reduced to mere churnalism.</p>
<p>There are two different kinds of evidence for this. One is the daily experience of almost every reporter at almost every level in almost every print and broadcast outlet in this country, captured, in a single example, by the young provincial reporter who kept a work journal for me [see diary, below].</p>
<p>In his working week, he turned out 48 stories – and yet he spent a total of only three hours out of the office and spoke face to face to only four people. That is churnalism.</p>
<p>No reporter who is producing nearly 10 stories every shift can possibly be doing their job properly. No reporter who spends nearly 95 per cent of the time crouched over a desk can possibly develop enough good leads or build enough good contacts. No reporter who speaks to so few people in researching 48 stories can possibly be checking their truth.</p>
<p>All local and regional media outlets in Britain &#8211; print and broadcast &#8211; have been swamped by a tide of churnalism. The scale and quality of coverage has been swept away. But the tide has not stopped in the provinces. The big national outlets can still support some real journalism, but here too, churnalism has swept through newsrooms.</p>
<p>And this is where the second kind of evidence comes in. I raised some money from the Rowntree Foundation and gave it to the journalism department at Cardiff University and asked them to research our national news coverage. In a long report, they came up with three groups of statistics which – I think – tell the tale of the decline and fall of Fleet Street.</p>
<p>First, they analysed the origins of every single home news story which was carried by the four quality papers (Times, Telegraph, Guardian, Independent) as well as the Daily Mail during two randomly chosen weeks last year. They ended up looking at more than 2,000 stories.</p>
<p>With the help of The Guardian newsdesk, they then attempted to capture all of the incoming material which had been passed on to reporters during those two weeks. Where there was still any doubt about the origin of stories, they interviewed reporters from the different newspapers and then tracked backwards to find their source material. What they found was that these stories were composed overwhelmingly of second-hand material.</p>
<p>Fifty four per cent of the stories contained clear signs of PR input – and bear in mind that is a conservative figure. It excludes the tabloids, which carry far more celebrity stories dominated by PR material. It excludes the Financial Times and the quality finance pages, where City reporting is flooded with PR. It excludes the quality feature pages, which are heavily influenced by PR from film companies, theatres, broadcasters and publishers; and the sports pages, where access to teams and stars is run by PR.</p>

<div class="advert"><MIDDLEOFSTORY/></div>
<p>In addition, the Cardiff researchers found that 70 per cent of these quality news stories were wholly or partly recycled from agency copy, usually the Press Association.</p>
<p>Now, if you merge the two figures – for PR and PA – and ask how much of Fleet Street quality news is, in fact, the work of Fleet Street&#8217;s own reporters, the answer from the Cardiff research is: 80 per cent of it is wholly, mainly or partially made up of second-hand material from PR and PA; eight per cent of </p>
<p>it was impossible to trace; and only 12 per cent of it was provably based on material generated by Fleet Street reporters. </p>
<p>The point here is not simply that we are recycling a mass of second-hand material, much of it specifically designed to serve somebody&#8217;s commercial or political interests. As a second area of research, the Cardiff specialists went on to look at those stories which relied on a specific statement of fact – and found that only 12 per cent of these stories showed any evidence that the central statement had been thoroughly checked.</p>
<p>Just pause and consider what those figures are suggesting: that only 12 per cent of the news stories in our most prestigious outlets are actually based entirely on material produced by the reporters who wrote them; and that only 12 per cent of the facts in those stories show evidence of having been properly checked. Of course, we run stories that aren&#8217;t true!</p>
<p>As the Cardiff researchers concluded: &#8216;Taken together, these data portray a picture of journalism in which any meaningful independent journalistic activity by the press is the exception rather than the rule. We are not talking about investigative journalism here, but the everyday practices of news judgement, fact-checking, balance, criticising and interrogating sources et cetera, that are, in theory, central to routine, day-to-day journalism.&#8221;</p>
<p>And then there is the third key statistic. I asked the Cardiff researchers to find all the information they could on two key points: the amount of space which Fleet Street papers have been filling with editorial copy; and the number of staff who have been hired to do so. They gathered this data for each of the past 20 years.</p>
<p>To check the space, they simply counted the pages in each paper for each year and subtracted the space that was used for advertisements. To check the staffing levels, they ploughed through annual reports for every Fleet Street company and, where necessary, spoke off-the-record to senior executives. </p>
<p>It should be said that there are some gaps in this information where companies failed to disclose figures for particular years; and some complications when Fleet Street companies bought non-national titles and included those figures in their total of employees and also failed to include freelance hiring. But the crude underlying trend which they found is overwhelming.</p>
<p>After 20 years which have seen a great many cuts and occasional bouts of new hiring, average staffing levels across Fleet Street companies are now slightly lower than they were two decades ago. But the amount of editorial space which those journalists are filling has trebled. </p>
<p>To put it another way, during those 20 years, the average time allowed for national newspaper journalists to find and check their stories has been cut to a third of its former level. </p>
<p>Even allowing for the role of journalists on freelance contracts who won&#8217;t show up in these figures, if you include their extra work on free sheets, websites, blogs, podcasts and vodcasts, you see that their time has been cut still further.</p>
<p>That is a disaster. That is why we have ended up recycling so much PR and wire copy, and that&#8217;s why we now effectively check only something like 12 per cent of our facts. That&#8217;s how our profession has been corrupted.</p>
<p>Next week: Nick Davies looks at the Press Association</p>
<p>Flat Earth News is published by Chatto and Windus on 7 February, priced £17.99</p>
                
                                        <div class="c-tags"> 
                             
        </div>
        <div data-spotim-module="conversation"
        data-post-id="40117"
        data-post-url="https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/"
        >
        </div><div class="c-share c-share--clear">
    <ul class="c-share__list">
        <li>
            <a href="" class="c-share__link icon-email" title="Share in an email">
            </a>
        </li>
        <li>
            <a class="c-share__link  icon-linkedin" href="http://www.linkedin.com/shareArticle?mini=true&url=https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/" target="_blank" title="Share on LinkedIN">
            </a>
        </li>
        <li>
            <a class="c-share__link  icon-twitter"  href="https://twitter.com/intent/tweet?url=https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/" target="_blank" title="Share on Twitter">
            </a>
        </li>
        <li>
            <a class="c-share__link  icon-facebook" href="https://www.facebook.com/sharer/sharer.php?u=https://pressgazette.co.uk/publishers/nationals/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117/" target="_blank" title="Share on facebook">
            </a>
        </li>
    </ul>
</div>

    <div class="c-podcast">
    <iframe src=https://embed.acast.com/623317507e51370012bc96c3?feed=true  frameBorder="0" width="100%" height="280px"></iframe>
</div>
<p>Email <b><a href = "mailto: pged@pressgazette.co.uk">pged@pressgazette.co.uk</a></b> to point out mistakes, provide story tips or send in a letter for publication on our "Letters Page" blog </p>                <div id="piano-paywall"></div>
                            </div>

        </section><!-- .entry-content /-->

        <aside id="sidebar-container" class="c-sidebar">
    <div class="c-googleAd-container">
    <div class="GAS_desctop GOOGLE_ADS_SLOT" data-sizes="300,600" style="display:none; overflow: hidden;margin-bottom: 36px;"></div>
</div>
    
         
        <!-- Jobbio div start -->
		<!--<div class="c-read-next" style="margin: 1rem auto; border-top: 1px solid #ccc; padding-top:1rem;">
            <div id='widget-display' class='jobbioapp'></div>
        
            <script>
                document.addEventListener('DOMContentLoaded', function () {
                jobbio_display.widget({
                    slug: 'press-gazette-jobs',
                    container: 'widget-display',
                    style: 'sidebar',
                    count: 3,
                    type: 'multiple',
                    content: 'jobs',
                    searchTerm: '',
                });
                }, false);
            </script>              
        </div> -->
     
</aside>



    </article><!-- #the-post /-->

    
    <div class="post-components">
        <div class="prj--rr">
		<div class="prj--rr-col">
        		</div>
</div>
    </div><!-- .post-components /-->

    
</div><!-- .main-content -->


<div class="footer-widgets-container">
    <div class="container">
        
        <footer class="c-article-footer">
            <div class="c-article-footer__container l-segment " data-columns=3>

                                <div class="l-segment__item">
                    <div class="c-author-related">
    
    <h4 class="c-author-related__title c-widget__title">More from this author
        <a href="https://pressgazette.co.uk/author/" class="c-widget__link--all">See All</a>    
     </h4>

    <div class="c-author-related__articles">
                     
            <div class="c-author-related__article">
            <a href="https://pressgazette.co.uk/news/news-diary-11-17-may-cannes-film-festival-trump-in-china-fa-cup-final/" class="c-author-related__img-link"> <img width="180" height="135" src="https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/cannesfilmfestival-180x135.jpg" class="attachment-post-thumbnail size-post-thumbnail wp-post-image" alt="Uma Thurman attends the &#039;Ismael&#039;s Ghosts (Les Fantomes d&#039;Ismael)&#039; screening and Opening Gala during the 70th annual Cannes Film Festival at Palais des Festivals on May 17, 2017 in Cannes, France. Picture: Denis Makarenko/Shutterstock" decoding="async" srcset="https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/cannesfilmfestival-180x135.jpg 180w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/cannesfilmfestival-525x394.jpg 525w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/cannesfilmfestival-800x600.jpg 800w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/cannesfilmfestival-200x150.jpg 200w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/cannesfilmfestival-1536x1152.jpg 1536w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/cannesfilmfestival-2048x1536.jpg 2048w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/cannesfilmfestival-397x298.jpg 397w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/cannesfilmfestival-314x235.jpg 314w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/cannesfilmfestival-464x348.jpg 464w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/cannesfilmfestival-735x551.jpg 735w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/cannesfilmfestival-1038x778.jpg 1038w" sizes="(max-width: 1407px) 1407px, (max-width: 335px) 335px, (max-width: 705px) 705px, (max-width: 335px) 335px, (max-width: 689px) 689px, (max-width: 336px) 336px, (max-width: 210px) 210px, (max-width: 101px) 101px, (max-width: 1024px) 1024px, (max-width: 101px) 101px, (max-width: 397px) 397px, (max-width: 464px) 464px, (max-width: 797px) 797px, (max-width: 960px) 960px, (max-width: 314px) 314px, (max-width: 464px) 464px, (max-width: 735px) 735px, (max-width: 1038px) 1038px" /> </a>         
            <div class="c-author-related__meta">
                                <a href="https://pressgazette.co.uk/author/foresight-news-2/" class="c-author-related__author-link">Foresight News </a>
                            <h6 class="c-author-related__headline"><a href="https://pressgazette.co.uk/news/news-diary-11-17-may-cannes-film-festival-trump-in-china-fa-cup-final/ ">News diary 11-17 May: Cannes Film Festival, Trump in China, FA Cup final </a></h6>
    </div>
            
            </div>
             
            <div class="c-author-related__article">
            <a href="https://pressgazette.co.uk/media-audience-and-business-data/media_metrics/most-popular-websites-news-world-monthly-2/" class="c-author-related__img-link"> <img width="180" height="135" src="https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/top504-180x135.jpg" class="attachment-post-thumbnail size-post-thumbnail wp-post-image" alt="Rediff, NDTV and Forbes were the only newsbrands to see month-on-month growth in April" decoding="async" srcset="https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/top504-180x135.jpg 180w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/top504-525x394.jpg 525w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/top504-800x600.jpg 800w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/top504-200x150.jpg 200w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/top504-1536x1152.jpg 1536w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/top504-2048x1536.jpg 2048w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/top504-397x298.jpg 397w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/top504-314x235.jpg 314w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/top504-464x348.jpg 464w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/top504-735x551.jpg 735w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/top504-1038x778.jpg 1038w" sizes="(max-width: 1407px) 1407px, (max-width: 335px) 335px, (max-width: 705px) 705px, (max-width: 335px) 335px, (max-width: 689px) 689px, (max-width: 336px) 336px, (max-width: 210px) 210px, (max-width: 101px) 101px, (max-width: 1024px) 1024px, (max-width: 101px) 101px, (max-width: 397px) 397px, (max-width: 464px) 464px, (max-width: 797px) 797px, (max-width: 960px) 960px, (max-width: 314px) 314px, (max-width: 464px) 464px, (max-width: 735px) 735px, (max-width: 1038px) 1038px" /> </a>         
            <div class="c-author-related__meta">
                                <a href="https://pressgazette.co.uk/author/pg_editor/" class="c-author-related__author-link">Press Gazette </a>
                            <h6 class="c-author-related__headline"><a href="https://pressgazette.co.uk/media-audience-and-business-data/media_metrics/most-popular-websites-news-world-monthly-2/ ">Top 50 English-language news sites in the world in April: Just three newsbrands grow traffic in past month </a></h6>
    </div>
            
            </div>
             
            <div class="c-author-related__article">
            <a href="https://pressgazette.co.uk/media-audience-and-business-data/media_metrics/most-popular-websites-news-us-monthly-3/" class="c-author-related__img-link"> <img width="180" height="135" src="https://pressgazette.co.uk/wp-content/uploads/sites/7/2024/11/newsweek-traffic-grows-scaled-e1730742743904-180x135.webp" class="attachment-post-thumbnail size-post-thumbnail wp-post-image" alt="A screenshot of the banner on the Newsweek website in 2018, illustrating a story about what&#039;s powered the magazine&#039;s rapid increase in web traffic and its attempts to build a more stable direct audience." decoding="async" loading="lazy" srcset="https://pressgazette.co.uk/wp-content/uploads/sites/7/2024/11/newsweek-traffic-grows-scaled-e1730742743904-180x135.webp 180w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2024/11/newsweek-traffic-grows-scaled-e1730742743904-526x394.webp 526w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2024/11/newsweek-traffic-grows-scaled-e1730742743904-800x599.webp 800w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2024/11/newsweek-traffic-grows-scaled-e1730742743904-200x150.webp 200w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2024/11/newsweek-traffic-grows-scaled-e1730742743904-1536x1151.webp 1536w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2024/11/newsweek-traffic-grows-scaled-e1730742743904-2048x1535.webp 2048w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2024/11/newsweek-traffic-grows-scaled-e1730742743904-397x297.webp 397w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2024/11/newsweek-traffic-grows-scaled-e1730742743904-314x235.webp 314w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2024/11/newsweek-traffic-grows-scaled-e1730742743904-464x348.webp 464w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2024/11/newsweek-traffic-grows-scaled-e1730742743904-735x551.webp 735w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2024/11/newsweek-traffic-grows-scaled-e1730742743904-1038x778.webp 1038w" sizes="(max-width: 1407px) 1407px, (max-width: 335px) 335px, (max-width: 705px) 705px, (max-width: 335px) 335px, (max-width: 689px) 689px, (max-width: 336px) 336px, (max-width: 210px) 210px, (max-width: 101px) 101px, (max-width: 1024px) 1024px, (max-width: 101px) 101px, (max-width: 397px) 397px, (max-width: 464px) 464px, (max-width: 797px) 797px, (max-width: 960px) 960px, (max-width: 314px) 314px, (max-width: 464px) 464px, (max-width: 735px) 735px, (max-width: 1038px) 1038px" /> </a>         
            <div class="c-author-related__meta">
                                <a href="https://pressgazette.co.uk/author/press-gazette/" class="c-author-related__author-link">Press Gazette </a>
                            <h6 class="c-author-related__headline"><a href="https://pressgazette.co.uk/media-audience-and-business-data/media_metrics/most-popular-websites-news-us-monthly-3/ ">Top 50 US news websites: Half of sites see traffic fall 20% or more in a year </a></h6>
    </div>
            
            </div>
           </div>
</div>
                </div>
                
                                <div class="l-segment__item">
                    <h3 class="c-widget__title">More of this topic</h3><div id="posts-list-widget-2" class="c-container-wrapper c-widget posts-list"><div class="posts-list-items">
<div class="c-af__related-article">
    
    
			<a aria-label="Wall Street Journal story on Trump and Epstein took six months and 20 staff" href="https://pressgazette.co.uk/media_law/wall-street-journal-story-on-trump-and-epstein-took-six-months-and-20-staff/" class="c-af__related__img-link"><img width="314" height="235" class="the-global-title" src="https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/emmatuckerjoshiherrmann-314x235.webp" alt="Wall Street Journal story on Trump and Epstein took six months and 20 staff" title="Wall Street Journal story on Trump and Epstein took six months and 20 staff" loading="lazy" srcset="https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/emmatuckerjoshiherrmann-314x235.webp 314w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/emmatuckerjoshiherrmann-525x394.webp 525w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/emmatuckerjoshiherrmann-800x600.webp 800w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/emmatuckerjoshiherrmann-200x150.webp 200w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/emmatuckerjoshiherrmann-397x298.webp 397w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/emmatuckerjoshiherrmann-180x135.webp 180w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/emmatuckerjoshiherrmann-464x348.webp 464w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/emmatuckerjoshiherrmann-735x551.webp 735w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/emmatuckerjoshiherrmann-1038x778.webp 1038w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/05/emmatuckerjoshiherrmann.webp 1200w" sizes="(max-width: 314px) 100vw, 314px" /></a>        <div class="c-af__meta">
        <a class="c-af__related__cat figaro-cat-15 the-global-title" href="https://pressgazette.co.uk/media_law/">Media Law</a>        <h6 class="c-af__related__headline"><a href="https://pressgazette.co.uk/media_law/wall-street-journal-story-on-trump-and-epstein-took-six-months-and-20-staff/"
            title="Wall Street Journal story on Trump and Epstein took six months and 20 staff">Wall Street Journal story on Trump and Epstein took six months and 20 staff</a></h6>
    </div>
    
</div>

<div class="c-af__related-article">
    
    
			<a aria-label="Gabriel Pogrund to lead expanded Sunday Times investigations team" href="https://pressgazette.co.uk/the-wire/media-jobs-uk-news/gabriel-pogrund-to-lead-expanded-sunday-times-investigations-team/" class="c-af__related__img-link"><img width="314" height="235" class="the-global-title" src="https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/insightteam-e1776956162102-314x235.webp" alt="Gabriel Pogrund to lead expanded Sunday Times investigations team" title="Gabriel Pogrund to lead expanded Sunday Times investigations team" loading="lazy" srcset="https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/insightteam-e1776956162102-314x235.webp 314w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/insightteam-e1776956162102-524x394.webp 524w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/insightteam-e1776956162102-800x601.webp 800w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/insightteam-e1776956162102-200x150.webp 200w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/insightteam-e1776956162102-397x298.webp 397w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/insightteam-e1776956162102-180x135.webp 180w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/insightteam-e1776956162102-464x348.webp 464w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/insightteam-e1776956162102-735x551.webp 735w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/04/insightteam-e1776956162102.webp 953w" sizes="(max-width: 314px) 100vw, 314px" /></a>        <div class="c-af__meta">
        <a class="c-af__related__cat figaro-cat-9422 the-global-title" href="https://pressgazette.co.uk/the-wire/media-jobs-uk-news/">Job Moves</a>        <h6 class="c-af__related__headline"><a href="https://pressgazette.co.uk/the-wire/media-jobs-uk-news/gabriel-pogrund-to-lead-expanded-sunday-times-investigations-team/"
            title="Gabriel Pogrund to lead expanded Sunday Times investigations team">Gabriel Pogrund to lead expanded Sunday Times investigations team</a></h6>
    </div>
    
</div>

<div class="c-af__related-article">
    
    
			<a aria-label="Sun merges features desk and Fabulous team" href="https://pressgazette.co.uk/publishers/nationals/jobs-at-risk-as-sun-set-to-merge-features-desk-and-fabulous-team/" class="c-af__related__img-link"><img width="314" height="235" class="the-global-title" src="https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/sunfabulous1-314x235.webp" alt="Sun merges features desk and Fabulous team" title="Sun merges features desk and Fabulous team" loading="lazy" srcset="https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/sunfabulous1-314x235.webp 314w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/sunfabulous1-525x394.webp 525w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/sunfabulous1-800x600.webp 800w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/sunfabulous1-200x150.webp 200w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/sunfabulous1-397x298.webp 397w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/sunfabulous1-180x135.webp 180w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/sunfabulous1-464x348.webp 464w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/sunfabulous1-735x551.webp 735w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/sunfabulous1-1038x778.webp 1038w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/sunfabulous1.webp 1280w" sizes="(max-width: 314px) 100vw, 314px" /></a>        <div class="c-af__meta">
        <a class="c-af__related__cat figaro-cat-1 the-global-title" href="https://pressgazette.co.uk/publishers/nationals/">National Newspapers</a>        <h6 class="c-af__related__headline"><a href="https://pressgazette.co.uk/publishers/nationals/jobs-at-risk-as-sun-set-to-merge-features-desk-and-fabulous-team/"
            title="Sun merges features desk and Fabulous team">Sun merges features desk and Fabulous team</a></h6>
    </div>
    
</div>
</div><div class="clearfix"></div></div><!-- .widget /-->                </div>
                
                                <div class="l-segment__item">
                    
<h3 class="c-widget__title">Recent Podcasts</h3>
<div class="posts-list-items">

<div class="c-af__related-article">
    
    
			<a aria-label="Reporting Andrew arrest, robot reporters at Mediahuis and Dom&#8217;s verdict on Prince Harry trial" href="https://pressgazette.co.uk/podcast-future-of-media-explained/reporting-andrew-arrest-robot-reporters-at-mediahuis-and-doms-verdict-on-prince-harry-trial/" class="c-af__related__img-link"><div class=" c-story__image--podcast"><img width="314" height="235" class="the-global-title" src="https://pressgazette.co.uk/wp-content/uploads/sites/7/2016/04/podcast_logo-e1653294846171.jpg" alt="Reporting Andrew arrest, robot reporters at Mediahuis and Dom&#039;s verdict on Prince Harry trial" title="Reporting Andrew arrest, robot reporters at Mediahuis and Dom&#039;s verdict on Prince Harry trial" loading="lazy" srcset="" sizes="(max-width: 314px) 100vw, 314px" /></div></a>        <div class="c-af__meta">
        <a class="c-af__related__cat figaro-cat-8575 the-global-title" href="https://pressgazette.co.uk/podcast-future-of-media-explained/">PG Podcast</a>        <h6 class="c-af__related__headline"><a href="https://pressgazette.co.uk/podcast-future-of-media-explained/reporting-andrew-arrest-robot-reporters-at-mediahuis-and-doms-verdict-on-prince-harry-trial/"
            title="Reporting Andrew arrest, robot reporters at Mediahuis and Dom&#8217;s verdict on Prince Harry trial">Reporting Andrew arrest, robot reporters at Mediahuis and Dom&#8217;s verdict on Prince Harry trial</a></h6>
    </div>
    
</div>

<div class="c-af__related-article">
    
    
			<a aria-label="Quality not discounts: How The Atlantic is powering subscriber growth" href="https://pressgazette.co.uk/podcast-future-of-media-explained/quality-not-discounts-how-the-atlantic-is-powering-subscriber-growth/" class="c-af__related__img-link"><div class=" c-story__image--podcast"><img width="314" height="235" class="the-global-title" src="https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/graham_megha-314x235.png" alt="Quality not discounts: How The Atlantic is powering subscriber growth" title="Quality not discounts: How The Atlantic is powering subscriber growth" loading="lazy" srcset="https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/graham_megha-314x235.png 314w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/graham_megha-200x150.png 200w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/graham_megha-464x348.png 464w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/graham_megha-735x551.png 735w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/02/graham_megha-1038x775.png 1038w" sizes="(max-width: 314px) 100vw, 314px" /></div></a>        <div class="c-af__meta">
        <a class="c-af__related__cat figaro-cat-8575 the-global-title" href="https://pressgazette.co.uk/podcast-future-of-media-explained/">PG Podcast</a>        <h6 class="c-af__related__headline"><a href="https://pressgazette.co.uk/podcast-future-of-media-explained/quality-not-discounts-how-the-atlantic-is-powering-subscriber-growth/"
            title="Quality not discounts: How The Atlantic is powering subscriber growth">Quality not discounts: How The Atlantic is powering subscriber growth</a></h6>
    </div>
    
</div>

<div class="c-af__related-article">
    
    
			<a aria-label="Semafor plans investment and expansion after celebrating first profitable year" href="https://pressgazette.co.uk/north-america/semafor-2026-ben-smith-interview/" class="c-af__related__img-link"><div class=" c-story__image--podcast"><img width="314" height="235" class="the-global-title" src="https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/01/ben_smith_color5-e1769532249612-314x235.webp" alt="Semafor plans investment and expansion after celebrating first profitable year" title="Semafor plans investment and expansion after celebrating first profitable year" loading="lazy" srcset="https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/01/ben_smith_color5-e1769532249612-314x235.webp 314w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/01/ben_smith_color5-e1769532249612-525x394.webp 525w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/01/ben_smith_color5-e1769532249612-800x601.webp 800w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/01/ben_smith_color5-e1769532249612-200x150.webp 200w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/01/ben_smith_color5-e1769532249612-1536x1153.webp 1536w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/01/ben_smith_color5-e1769532249612-397x298.webp 397w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/01/ben_smith_color5-e1769532249612-180x135.webp 180w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/01/ben_smith_color5-e1769532249612-464x348.webp 464w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/01/ben_smith_color5-e1769532249612-735x551.webp 735w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/01/ben_smith_color5-e1769532249612-1038x778.webp 1038w, https://pressgazette.co.uk/wp-content/uploads/sites/7/2026/01/ben_smith_color5-e1769532249612.webp 1710w" sizes="(max-width: 314px) 100vw, 314px" /></div></a>        <div class="c-af__meta">
        <a class="c-af__related__cat figaro-cat-8584 the-global-title" href="https://pressgazette.co.uk/north-america/">North America</a>        <h6 class="c-af__related__headline"><a href="https://pressgazette.co.uk/north-america/semafor-2026-ben-smith-interview/"
            title="Semafor plans investment and expansion after celebrating first profitable year">Semafor plans investment and expansion after celebrating first profitable year</a></h6>
    </div>
    
</div>
</div><div class="clearfix"></div></div><!-- .widget /-->                </div>
                
                
        </footer><!-- .footer-widget-area /-->

                </div><!-- .container /-->
</div><!-- #Footer-widgets-container /-->
        
</div><!-- /.wrapper__inner -->
</div><!-- /.wrapper -->
<a id="scrollToTopButton" href="#"><i class="icon-chevron-up"></i></a>
</main><!-- /.main -->
    <footer class="c-footer">
        <div class="c-footer__container">
                        <div class="c-footer__nav">
                <div class="c-footer__title">Press Gazette</div>
                <div class="menu-footer-site-links-container"><ul id="menu-footer-site-links" class="c-footer-menu"><li id="menu-item-193550" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-193550"><a href="https://pressgazette.co.uk/about/">About us</a></li>
<li id="menu-item-193553" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-home menu-item-193553"><a href="https://pressgazette.co.uk/">Home Page</a></li>
<li id="menu-item-193555" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-193555"><a href="https://pressgazette.co.uk/contact/">Contact us</a></li>
<li id="menu-item-217186" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-217186"><a href="https://pressgazette.co.uk/advertise-on-press-gazette/">Advertising &#038; Partnerships</a></li>
</ul></div>            </div>
            <div class="c-footer__social">
                <div class="c-footer__title">Social</div>
                <div class="c-social">
                    <ul><li class="c-social-icons-item"><a class="c-social-link facebook-social-icon" rel="external noopener nofollow" target="_blank" href="https://www.facebook.com/pressgazette" title="Follow us on Facebook"><span class="icon-facebook"></span><span class="screen-reader-text">Facebook</span></a></li><li class="c-social-icons-item"><a class="c-social-link twitter-social-icon" rel="external noopener nofollow" target="_blank" href="https://twitter.com/pressgazette" title="Follow us on Twitter"><span class="icon-twitter"></span><span class="screen-reader-text">Twitter</span></a></li></ul> 
                </div>
            </div>
                            <div class="c-footer__legal">
                    <div class="c-footer__title">Legals</div>
                    <div class="c-footer__menu"><ul id="menu-legals" class="c-footer-menu"><li id="menu-item-193559" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-193559"><a href="https://pressgazette.co.uk/editorial-standards/">Editorial standards/Complaints and Corrections</a></li>
<li id="menu-item-193646" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-193646"><a href="https://pressgazette.co.uk/privacy-policy/">Privacy Policy</a></li>
<li id="menu-item-193647" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-193647"><a href="https://pressgazette.co.uk/terms-conditions/">Terms and Conditions</a></li>
</ul></div>                </div>
                        <div class="c-footer__copyright">
                <div class="c-copyright-text c-copyright-text-first">Fighting for quality news media in the digital age.</div><div class="c-copyright-text c-copyright-text-second">© Copyright 2026 PROGRESSIVE MEDIA INVESTMENTS LTD.</div>            </div>

                                </div>
    </footer><!-- /.footer -->
<script>
// GOOGLE SCRIPT

let allADDslots = document.getElementsByClassName("GOOGLE_ADS_SLOT");
let mobileWH = 768, tabletWH = 1024, currentDocumentWidth = 0;

function getTheDocumentWidth() {
    let DW = document.documentElement.clientWidth;
    currentDocumentWidth = DW;
    return DW
}

function showHideHeadBanners(){
    findCurentView();
    console.log(curentView);
    let activeBannersList = [];

    Array.prototype.forEach.call(allADDslots, function (allADDslots, index) {
        allADDslots.style.display = "none";
    });

    let currentView = "GAS_" + curentView;
    let vissibleSlots = document.getElementsByClassName(currentView);

    Array.prototype.forEach.call(vissibleSlots, function (vissibleSlots, index) {
        vissibleSlots.style.display = "block";
        activeBannersList.push(vissibleSlots.getAttribute('id'))
    });
    activeBannersList.forEach((element, index) => { 
        let curentObject = document.getElementById(element);  
        let bannerStatus = curentObject.getAttribute("data-bannerStatus");
        if(bannerStatus != "1"){
console.log("trigerit")
        googletag.cmd.push(function () { googletag.display(element); });
        curentObject.setAttribute("data-bannerStatus","1");
        }
        //DO NOT RELOAD !!!!!
    }); 
}

function findCurentView() {
    if (getTheDocumentWidth() > 0) {
        curentView = "mobile";
        if (getTheDocumentWidth() > mobileWH) {
            curentView = "tablet";
            if (getTheDocumentWidth() > tabletWH) {
                curentView = "desctop";
            }
        }
    }
    return curentView;
}

window.addEventListener("resize", showHideHeadBanners);


function convertToAnArray(data) {
    let testTHing = data;
    let newArray = [];
    if (testTHing.includes("-")) {
        var nameArr = testTHing.split('-');
        nameArr.forEach((element, index) => {
            newArray.push(element.split(",").map(Number));
        });
    } else {
        // newArray[0] = [];
        newArray.push(testTHing.split(",").map(Number));
    }
    return newArray;
}



googletag.cmd.push(function () {
    let createDATA = [];
    Array.prototype.forEach.call(allADDslots, function (allADDslots, index) {
        let googleIdentity = '/' + googleSessionID + '/' + googleProjectName + '/' + googleProjectSession + '_Unit' + (index + 1);
        let curentSlotSizes = allADDslots.getAttribute('data-sizes');
        let curentSlotID = 'div-gpt-ad-' + googleAdsID + '-' + (index + 1); // allADDslots.getAttribute('id');
        allADDslots.setAttribute('id', curentSlotID);


        //data-ifStorefront-size

        let additionalSizes;

        if(document.querySelector('body').classList.contains('storefronts-template-default') && allADDslots.getAttribute('data-ifStorefront-size')){
            curentSlotSizes = allADDslots.getAttribute('data-ifStorefront-size');
            console.log("SF custom size");
            console.log(curentSlotSizes);
        }

        createDATA[index] = [googleIdentity, curentSlotSizes, curentSlotID];
    });

    createDATA.forEach((element, index) => {
        googletag.defineSlot(element[0], convertToAnArray(element[1]), element[2]).addService(googletag.pubads());
    });
    
   // googletag.pubads().enableSingleRequest(); //trigger to load all banners instantly
    googletag.enableServices();
    showHideHeadBanners();
});


</script><script type="speculationrules">
{"prefetch":[{"source":"document","where":{"and":[{"href_matches":"\/*"},{"not":{"href_matches":["\/wp-*.php","\/wp-admin\/*","\/wp-content\/uploads\/sites\/7\/*","\/wp-content\/*","\/wp-content\/plugins\/*","\/wp-content\/themes\/pressgazette\/*","\/wp-content\/themes\/figaro\/*","\/*\\?(.+)"]}},{"not":{"selector_matches":"a[rel~=\"nofollow\"]"}},{"not":{"selector_matches":".no-prefetch, .no-prefetch a"}}]},"eagerness":"conservative"}]}
</script>
        <section class="afterFooter" id="nsmgfooterlogoscompanies">
    <div class="afterFooter__container" >
        <div class="afterFooter__headLine">
            Websites in our network        </div>
        <div class="afterFooter__companyList --imgList">            
                    <a
         href="https://www.newstatesman.com/events" rel="nofollow"        >
            <img
             alt="https://www.newstatesman.com/events" src="https://live-newstatesman-multisite.pantheonsite.io/pressgazette/wp-content/uploads/sites/7/2022/10/NS.svg" width="130" height="32"    >
        </a>
                        </div>  
        <!-- end if IMG buttons -->  
    </div>
</section>
                     <script>
        jQuery('.c-related-companies-article__single').hover(function() { 
        var hoverTime = 1000;
        var elements = [].slice.apply(document.querySelectorAll('.c-related-companies-article__single'));
        elements.forEach(function (el) {
        var timeout;
       el.addEventListener('mouseover', function (){
        var match = jQuery(this).attr('data-url');
        var hovertitle = jQuery(this).attr('data-target');
        var curpostid = jQuery(this).attr('id');  
        var getpostid = curpostid.replace('company', '');
                if(!jQuery(this).hasClass("verify")){
            jQuery(this).addClass("verify");
            timeout = window.setTimeout(function () {
                permutive.track('TabClick', {
                    article: {
                        url: match,               
                        title:hovertitle
                    }
                });
            }, hoverTime);
        }
      });
      
     });
    
    });
    
    </script>
        <!-- This site is converting visitors into subscribers and customers with OptinMonster - https://optinmonster.com :: Campaign Title: PG - Before you go.... -->
<script>(function(d,u,ac){var s=d.createElement('script');s.type='text/javascript';s.src='https://a.omappapi.com/app/js/api.min.js';s.async=true;s.dataset.user=u;s.dataset.campaign=ac;d.getElementsByTagName('head')[0].appendChild(s);})(document,183417,'drzvd3jgrr4wq0oorrt6');</script>
<!-- / OptinMonster -->		<script type="text/javascript">
		var drzvd3jgrr4wq0oorrt6_shortcode = true;		</script>
		<script type="text/javascript" src="https://www.google.com/recaptcha/api.js?ver=1.0" id="ha-recaptcha-js"></script>
<script type="text/javascript" id="nsmgscampaigns-js-extra">
/* <![CDATA[ */
var nsmgscampaigns = {"ajaxurl":"https:\/\/pressgazette.co.uk\/wp-admin\/admin-ajax.php","nonce":"d600970dfa"};
/* ]]> */
</script>
<script type="text/javascript" src="https://pressgazette.co.uk/wp-content/plugins/nsmg-campaigns/assets/public/js/main.min.js?ver=2.9#defer" id="nsmgscampaigns-js"></script>
<script type="text/javascript" src="https://pressgazette.co.uk/wp-content/plugins/nsmg-footer-logos//assets/frontend.min.js?ver=1.0" id="nsmgfooterlogosfrontendJs-js"></script>
<script type="text/javascript" id="datalayer-script-js-extra">
/* <![CDATA[ */
var datalayer_script_vars = {"contentType":"post","pageName":"Nick Davies: 'Churnalism has taken the place of what we should be doing: Telling the truth'","metaTitle":"Nick Davies: 'Churnalism has taken the place of what we should be doing: Telling the truth'","pageId":"40117","storyType":"","domain":"https:\/\/pressgazette.co.uk","platform":"website","localDate":"04th Feb 2008","navigationDestinationPage":"nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117","NavigationLinkType":"post","pageUrl":"https:\/\/pressgazette.co.uk\/publishers\/nationals\/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117\/","pageUri":"\/publishers\/nationals\/nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117\/","utmSource":null,"lastModified":"2008-01-30 16:06:48[BST]","firstPublished":"2008-02-04 23:59:00[BST]","authorName":"","sponsored":"false","clientName":null,"editorialChannel":"","contentTags":null,"contentCategories":"National Newspapers,Regional Newspapers","postFormat":"","timeSincePublication":"576239525","articleHeadline":"Nick Davies: 'Churnalism has taken the place of what we should be doing: Telling the truth'","urlSlug":"nick-davies-churnalism-has-taken-the-place-of-what-we-should-be-doing-telling-the-truth-40117","metaDescription":"A lot of people think that British journalism is corrupted. I agree with them. Our job as journalists is to tell the truth, but repeatedly we fail. From the great global falsehoods on weapons of mass destruction and millennium bugs to the daily dribble of routine disinformation and distortion, we serve up stories which are \u2026","siteSection":null,"siteSubSection":null};
/* ]]> */
</script>
<script type="text/javascript" src="https://pressgazette.co.uk/wp-content/plugins/nsmg-tracking/js/datalayer.js?ver=1.0.6" id="datalayer-script-js"></script>
<script type="text/javascript" id="crb-google-tagmanager-js-extra">
/* <![CDATA[ */
var crbTracker = {"googleTagManager":{"id":"GTM-PFS4RD8"}};
/* ]]> */
</script>
<script type="text/javascript" src="https://pressgazette.co.uk/wp-content/plugins/nsmg-tracking//js/googleTagManager.js?ver=1778305208" id="crb-google-tagmanager-js"></script>
<script type="text/javascript" id="crb-permutive-js-extra">
/* <![CDATA[ */
var crbTracker = {"permutive":{"workspaceID":"5b476c7c-253b-4e23-9993-4df0b853b921","publicAPIKey":"50de2fe6-d235-41ef-b65a-3aef7af8d446","postTitle":"Nick Davies: &#8216;Churnalism has taken the place of what we should be doing: Telling the truth&#8217;","postID":40117,"postType":"post","postDate":"February 4, 2008","postAuthor":"","siteURL":"https:\/\/pressgazette.co.uk","wealthUser":false,"postTags":false,"postCategories":"National Newspapers, Regional Newspapers","collectDomains":"https:\/\/pressgazette.co.uk\/wp-content\/plugins\/nsmg-tracking\/\/js\/pollsInFrame.js","pageViewEvent":{"company_name":"","post_title":"Nick Davies: &#8216;Churnalism has taken the place of what we should be doing: Telling the truth&#8217;","post_id":40117,"categories":["National Newspapers","Regional Newspapers"],"tags":[],"author":"","post_type":"post","article_date":"2008-02-04T23:59:00Z","regions":[],"industry_taxonomy":[],"sectors":[],"report_type":""}}};
/* ]]> */
</script>
<script type="text/javascript" src="https://pressgazette.co.uk/wp-content/plugins/nsmg-tracking//js/permutive.js?ver=1778305208" id="crb-permutive-js"></script>
<script type="text/javascript" src="https://pressgazette.co.uk/wp-content/plugins/nsmg-tracking//js/permutiveData.min.js?ver=1.0.6" id="crb-permutive-data-js"></script>
<script type="text/javascript" src="https://pressgazette.co.uk/wp-content/themes/pressgazette/assets/vendor/swiper/swiper.min.js?ver=6.8.3" id="swiper-slider-js"></script>
<script type="text/javascript" src="https://loader.bridged.media/bridged-agent.js" id="bridged-loader-js"></script>
<script type="text/javascript" src="https://cdn.insurads.com/bootstrap/DHBWW4DB.js" id="insurads-script-js"></script>
<script type="text/javascript" id="figaro-scripts-js-extra">
/* <![CDATA[ */
var figaro = {"is_rtl":"","ajaxurl":"https:\/\/pressgazette.co.uk\/wp-admin\/admin-ajax.php","post_id":"40117","lazyload":"","ajax_loader":"<div class=\"loader-overlay\"><div class=\"spinner-circle\"><\/div><\/div>","type_to_search":""};
/* ]]> */
</script>
<script type="text/javascript" src="https://pressgazette.co.uk/code/wp-content/uploads/sites/7/figaro/assets/nsjs/figaro-main.js?v=1773201198&amp;ver=1773201198" id="figaro-scripts-js"></script>
<script type="text/javascript" src="https://pressgazette.co.uk/wp-content/themes/figaro/assets/dist/js/script.min.js?ver=1.124" id="company-script-js"></script>
<script>(function(d){var s=d.createElement("script");s.type="text/javascript";s.src="https://a.omappapi.com/app/js/api.min.js";s.async=true;s.id="omapi-script";d.getElementsByTagName("head")[0].appendChild(s);})(document);</script><script type="text/javascript" src="https://pressgazette.co.uk/wp-content/plugins/storefronts/assets/js/app.min.js?ver=5.7.6" id="main-js-js"></script>
<script type="text/javascript" src="https://pressgazette.co.uk/wp-content/plugins/optinmonster/assets/dist/js/helper.min.js?ver=2.10.0" id="optinmonster-wp-helper-js"></script>
		<script type="text/javascript">var omapi_localized = {
			ajax: 'https://pressgazette.co.uk/wp-admin/admin-ajax.php?optin-monster-ajax-route=1',
			nonce: '21f040af07',
			slugs:
			{"drzvd3jgrr4wq0oorrt6":{"slug":"drzvd3jgrr4wq0oorrt6","mailpoet":false}}		};</script>
				<script type="text/javascript">var omapi_data = {"wc_cart":[],"object_id":40117,"object_key":"post","object_type":"post","term_ids":[1,2],"wp_json":"https:\/\/pressgazette.co.uk\/wp-json"};</script>
		
<script async>
    const userSignInBtn = document.querySelector('#piano-signIn');
    const userSubscribeBtn = document.querySelector('#piano-subscribe');
    let loginButtons;

    if (userSignInBtn && userSubscribeBtn) {
        loginButtons = [userSignInBtn, userSubscribeBtn];
    } 

    loginButtons.forEach((loginButton) => {

        loginButton.addEventListener('click', () => {

            const articleRedirectUrl = window.location.href;

            if(window.location.href.includes('login') || window.location.href.includes('account')){
                localStorage.setItem('articleRedirectUrl', window.location.origin);    
            } else{
                localStorage.setItem('articleRedirectUrl', articleRedirectUrl);
            }

            

        });


    });
</script>

<script>
function PianoLogin() {
    console.log('This is signin screen');
    tp = window.tp || [];
    tp.push(["init", function () {
        tp.pianoId.show({
            disableSignUp: false,
            displayMode: 'modal',
            screen: 'login',
            //width:'800',
            //containerSelector: '#login-form',
            loggedIn: function (data) {
                console.log('user ', data.user, ' logged in with token', data.token);

                //redirect user to my account page after signin
                window.location = "/my-account";

            },

            loggedOut: function () {
                console.log('user logged out');
                //Set the CSS and HTML here for what the signup button should look like when the user is logged out //e.g. unhide the sign in button, hide the signout button

            }
        });
        
    }]);
}

</script>


<script type="text/javascript">window.NREUM||(NREUM={});NREUM.info={"beacon":"bam.nr-data.net","licenseKey":"NRJS-59243095c956cb51a60","applicationID":"571186038","transactionName":"ZgRUY0RSChcDU0BYXF9Od1RCWgsKTUNdX1RdBA==","queueTime":0,"applicationTime":818,"atts":"SkNXFQxIGRk=","errorBeacon":"bam.nr-data.net","agent":""}</script></body>

</html>
<!-- plugin=object-cache-pro client=phpredis metric#hits=6534 metric#misses=62 metric#hit-ratio=99.1 metric#bytes=20959510 metric#prefetches=9 metric#store-reads=175 metric#store-writes=15 metric#store-hits=325 metric#store-misses=46 metric#sql-queries=162 metric#ms-total=818.11 metric#ms-cache=224.19 metric#ms-cache-avg=1.1862 metric#ms-cache-ratio=27.4 -->

<!-- URL: https://news.gsu.edu/ -->
<!-- HTTP: 200 SIZE: 361837 -->
<!DOCTYPE html>
<html lang="en-US">
<head >
<meta charset="UTF-8" /><script type="text/javascript">(window.NREUM||(NREUM={})).init={ajax:{deny_list:["bam.nr-data.net"]},feature_flags:["soft_nav"]};(window.NREUM||(NREUM={})).loader_config={xpid:"VQYAWVZQCRADVVlSBAMDU1E=",licenseKey:"1be6db1351",applicationID:"318032455",browserID:"318032465"};;/*! For license information please see nr-loader-spa-1.314.0.min.js.LICENSE.txt */
(()=>{var e,t,r={384:(e,t,r)=>{"use strict";r.d(t,{NT:()=>a,Zm:()=>c,bQ:()=>u,dV:()=>d,pV:()=>l});var n=r(6154),i=r(1863),s=r(944),o=r(1910);const a={beacon:"bam.nr-data.net",errorBeacon:"bam.nr-data.net"};function c(){return n.gm.NREUM||(n.gm.NREUM={}),void 0===n.gm.newrelic&&(n.gm.newrelic=n.gm.NREUM),n.gm.NREUM}function d(){let e=c();return e.o||(e.o={ST:n.gm.setTimeout,SI:n.gm.setImmediate||n.gm.setInterval,CT:n.gm.clearTimeout,XHR:n.gm.XMLHttpRequest,REQ:n.gm.Request,EV:n.gm.Event,PR:n.gm.Promise,MO:n.gm.MutationObserver,FETCH:n.gm.fetch,WS:n.gm.WebSocket},(0,o.i)(...Object.values(e.o))),e}function u(e,t){let r=c();r.initializedAgents??={},t.initializedAt={ms:(0,i.t)(),date:new Date},r.initializedAgents[e]=t,2===Object.keys(r.initializedAgents).length&&(0,s.R)(69)}function l(){return function(){let e=c();const t=e.info||{};e.info={beacon:a.beacon,errorBeacon:a.errorBeacon,...t}}(),function(){let e=c();const t=e.init||{};e.init={...t}}(),d(),function(){let e=c();const t=e.loader_config||{};e.loader_config={...t}}(),c()}},782:(e,t,r)=>{"use strict";r.d(t,{T:()=>n});const n=r(860).K7.pageViewTiming},860:(e,t,r)=>{"use strict";r.d(t,{$J:()=>u,K7:()=>c,P3:()=>d,XX:()=>i,Yy:()=>a,df:()=>s,qY:()=>n,v4:()=>o});const n="events",i="jserrors",s="browser/blobs",o="rum",a="browser/logs",c={ajax:"ajax",genericEvents:"generic_events",jserrors:i,logging:"logging",metrics:"metrics",pageAction:"page_action",pageViewEvent:"page_view_event",pageViewTiming:"page_view_timing",sessionReplay:"session_replay",sessionTrace:"session_trace",softNav:"soft_navigations"},d={[c.pageViewEvent]:1,[c.pageViewTiming]:2,[c.metrics]:3,[c.jserrors]:4,[c.softNav]:5,[c.ajax]:6,[c.sessionTrace]:7,[c.sessionReplay]:8,[c.logging]:9,[c.genericEvents]:10},u={[c.pageViewEvent]:o,[c.pageViewTiming]:n,[c.ajax]:n,[c.softNav]:n,[c.metrics]:i,[c.jserrors]:i,[c.sessionTrace]:s,[c.sessionReplay]:s,[c.logging]:a,[c.genericEvents]:"ins"}},944:(e,t,r)=>{"use strict";r.d(t,{R:()=>i});var n=r(3241);function i(e,t){"function"==typeof console.debug&&(console.debug("New Relic Warning: https://github.com/newrelic/newrelic-browser-agent/blob/main/docs/warning-codes.md#".concat(e),t),(0,n.W)({drained:null,type:"data",name:"warn",feature:"warn",data:{code:e,secondary:t}}))}},993:(e,t,r)=>{"use strict";r.d(t,{A$:()=>s,ET:()=>o,TZ:()=>a,p_:()=>i});var n=r(860);const i={ERROR:"ERROR",WARN:"WARN",INFO:"INFO",DEBUG:"DEBUG",TRACE:"TRACE"},s={OFF:0,ERROR:1,WARN:2,INFO:3,DEBUG:4,TRACE:5},o="log",a=n.K7.logging},1687:(e,t,r)=>{"use strict";r.d(t,{Ak:()=>a,Ze:()=>d,x3:()=>c});var n=r(3241),i=r(3606),s=r(860),o=r(2646);function a(e,t){if(!e)return;const r={staged:!1,priority:s.P3[t]||0};e.runtime.drainRegistry.get(t)||e.runtime.drainRegistry.set(t,r)}function c(e,t){if(!e)return;const r=e.runtime.drainRegistry;r&&(r.get(t)&&r.delete(t),l(e,t,!1),r.size&&u(e))}function d(e,t="feature",r=!1){if(e){if(!e.runtime.drainRegistry.get(t)||r)return l(e,t);e.runtime.drainRegistry.get(t).staged=!0,u(e)}}function u(e){if(!e)return;const t=Array.from(e.runtime.drainRegistry);t.every(([e,t])=>t.staged)&&(t.sort((e,t)=>e[1].priority-t[1].priority),t.forEach(([t])=>{e.runtime.drainRegistry.delete(t),l(e,t)}))}function l(e,t,r=!0){if(!e)return;const s=e.ee,a=i.i.handlers;if(s&&!s.aborted&&s.backlog&&a){if((0,n.W)({type:"lifecycle",name:"drain",feature:t}),r){const e=s.backlog[t],r=a[t];if(r){for(let t=0;e&&t<e.length;++t)f(e[t],r);Object.entries(r).forEach(([e,t])=>{Object.values(t||{}).forEach(t=>{t[0]?.on&&t[0].context()instanceof o.y&&!t[0].listeners(e).includes(t[1])&&t[0].on(e,t[1])})})}}s.isolatedBacklog||delete a[t],s.backlog[t]=null,s.emit("drain-"+t,[])}}function f(e,t){var r=e[1];Object.values(t[r]||{}).forEach(t=>{var r=e[0];if(t[0]===r){var n=t[1],i=e[3],s=e[2];n.apply(i,s)}})}},1738:(e,t,r)=>{"use strict";r.d(t,{U:()=>f,Y:()=>l});var n=r(3241),i=r(9908),s=r(1863),o=r(944),a=r(3969),c=r(8362),d=r(860),u=r(4261);function l(e,t,r,s){const l=s||r;!l||l[e]&&l[e]!==c.d.prototype[e]||(l[e]=function(){(0,i.p)(a.xV,["API/"+e+"/called"],void 0,d.K7.metrics,r.ee),(0,n.W)({drained:!!r.runtime?.activatedFeatures,type:"data",name:"api",feature:u.Pl+e,data:{}});try{return t.apply(this,arguments)}catch(e){(0,o.R)(23,e)}})}function f(e,t,r,n,o){const a=e.info;null===r?delete a.jsAttributes[t]:a.jsAttributes[t]=r,(o||null===r)&&(0,i.p)(u.Pl+n,[(0,s.t)(),t,r],void 0,"session",e.ee)}},1741:(e,t,r)=>{"use strict";r.d(t,{W:()=>s});var n=r(944),i=r(4261);class s{#e(e,...t){if(this[e]!==s.prototype[e])return this[e](...t);(0,n.R)(35,e)}addPageAction(e,t){return this.#e(i.hG,e,t)}register(e){return this.#e(i.eY,e)}recordCustomEvent(e,t){return this.#e(i.fF,e,t)}setPageViewName(e,t){return this.#e(i.Fw,e,t)}setCustomAttribute(e,t,r){return this.#e(i.cD,e,t,r)}noticeError(e,t){return this.#e(i.o5,e,t)}setUserId(e,t=!1){return this.#e(i.Dl,e,t)}setApplicationVersion(e){return this.#e(i.nb,e)}setErrorHandler(e){return this.#e(i.bt,e)}addRelease(e,t){return this.#e(i.k6,e,t)}log(e,t){return this.#e(i.$9,e,t)}start(){return this.#e(i.d3)}finished(e){return this.#e(i.BL,e)}recordReplay(){return this.#e(i.CH)}pauseReplay(){return this.#e(i.Tb)}addToTrace(e){return this.#e(i.U2,e)}setCurrentRouteName(e){return this.#e(i.PA,e)}interaction(e){return this.#e(i.dT,e)}wrapLogger(e,t,r){return this.#e(i.Wb,e,t,r)}measure(e,t){return this.#e(i.V1,e,t)}consent(e){return this.#e(i.Pv,e)}}},1863:(e,t,r)=>{"use strict";function n(){return Math.floor(performance.now())}r.d(t,{t:()=>n})},1910:(e,t,r)=>{"use strict";r.d(t,{i:()=>s});var n=r(944);const i=new Map;function s(...e){return e.every(e=>{if(i.has(e))return i.get(e);const t="function"==typeof e?e.toString():"",r=t.includes("[native code]"),s=t.includes("nrWrapper");return r||s||(0,n.R)(64,e?.name||t),i.set(e,r),r})}},2555:(e,t,r)=>{"use strict";r.d(t,{D:()=>a,f:()=>o});var n=r(384),i=r(8122);const s={beacon:n.NT.beacon,errorBeacon:n.NT.errorBeacon,licenseKey:void 0,applicationID:void 0,sa:void 0,queueTime:void 0,applicationTime:void 0,ttGuid:void 0,user:void 0,account:void 0,product:void 0,extra:void 0,jsAttributes:{},userAttributes:void 0,atts:void 0,transactionName:void 0,tNamePlain:void 0};function o(e){try{return!!e.licenseKey&&!!e.errorBeacon&&!!e.applicationID}catch(e){return!1}}const a=e=>(0,i.a)(e,s)},2614:(e,t,r)=>{"use strict";r.d(t,{BB:()=>o,H3:()=>n,g:()=>d,iL:()=>c,tS:()=>a,uh:()=>i,wk:()=>s});const n="NRBA",i="SESSION",s=144e5,o=18e5,a={STARTED:"session-started",PAUSE:"session-pause",RESET:"session-reset",RESUME:"session-resume",UPDATE:"session-update"},c={SAME_TAB:"same-tab",CROSS_TAB:"cross-tab"},d={OFF:0,FULL:1,ERROR:2}},2646:(e,t,r)=>{"use strict";r.d(t,{y:()=>n});class n{constructor(e){this.contextId=e}}},2843:(e,t,r)=>{"use strict";r.d(t,{G:()=>s,u:()=>i});var n=r(3878);function i(e,t=!1,r,i){(0,n.DD)("visibilitychange",function(){if(t)return void("hidden"===document.visibilityState&&e());e(document.visibilityState)},r,i)}function s(e,t,r){(0,n.sp)("pagehide",e,t,r)}},3241:(e,t,r)=>{"use strict";r.d(t,{W:()=>s});var n=r(6154);const i="newrelic";function s(e={}){try{n.gm.dispatchEvent(new CustomEvent(i,{detail:e}))}catch(e){}}},3304:(e,t,r)=>{"use strict";r.d(t,{A:()=>s});var n=r(7836);const i=()=>{const e=new WeakSet;return(t,r)=>{if("object"==typeof r&&null!==r){if(e.has(r))return;e.add(r)}return r}};function s(e){try{return JSON.stringify(e,i())??""}catch(e){try{n.ee.emit("internal-error",[e])}catch(e){}return""}}},3333:(e,t,r)=>{"use strict";r.d(t,{$v:()=>u,TZ:()=>n,Xh:()=>c,Zp:()=>i,kd:()=>d,mq:()=>a,nf:()=>o,qN:()=>s});const n=r(860).K7.genericEvents,i=["auxclick","click","copy","keydown","paste","scrollend"],s=["focus","blur"],o=4,a=1e3,c=2e3,d=["PageAction","UserAction","BrowserPerformance"],u={RESOURCES:"experimental.resources",REGISTER:"register"}},3434:(e,t,r)=>{"use strict";r.d(t,{Jt:()=>o,YM:()=>u});var n=r(7836),i=r(5607),s=r(5732);const o="nr@original:".concat(i.W),a=50;var c=Object.prototype.hasOwnProperty,d=!1;function u(e,t,r){return e||(e=n.ee),i.inPlace=function(e,t,r,n,s,o){r||(r="");const a="-"===r.charAt(0);for(let c=0;c<t.length;c++){const d=t[c],u=e[d];f(u)||(e[d]=i(u,a?d+r:r,n,d,s,o))}},i.flag=o,i;function i(t,n,i,d,h,p){return f(t)?t:(n||(n=""),nrWrapper[o]=t,function(e,t,r){if(Object.defineProperty&&Object.keys)try{return Object.keys(e).forEach(function(r){Object.defineProperty(t,r,{get:function(){return e[r]},set:function(t){return e[r]=t,t}})}),t}catch(e){l([e],r)}for(var n in e)c.call(e,n)&&(t[n]=e[n])}(t,nrWrapper,e),nrWrapper);function nrWrapper(){var o,c,f,g;let m,v;try{c=this,o=[...arguments],v=p?(0,s.$5)(r):[void 0],f="function"==typeof i?i(o,c):i||{}}catch(t){l([t,"",[o,c,d],f],e)}u(n+"start",[o,c,d,v],f,h);const y=performance.now();let b;try{return g=t.apply(c,o),b=performance.now(),g}catch(e){throw b=performance.now(),u(n+"err",[o,c,e,v],f,h),m=e,m}finally{const e=b-y,t={start:y,end:b,duration:e,isLongTask:e>=a,methodName:d,thrownError:m};t.isLongTask&&u("long-task",[t,c,v],f,h),u(n+"end",[o,c,g,v],f,h)}}}function u(r,n,i,s){if(!d||t){var o=d;d=!0;try{e.emit(r,n,i,t,s)}catch(t){l([t,r,n,i],e)}d=o}}}function l(e,t){t||(t=n.ee);try{t.emit("internal-error",e)}catch(e){}}function f(e){return!(e&&"function"==typeof e&&e.apply&&!e[o])}},3606:(e,t,r)=>{"use strict";r.d(t,{i:()=>s});var n=r(9908);s.on=o;var i=s.handlers={};function s(e,t,r,s){o(s||n.d,i,e,t,r)}function o(e,t,r,i,s){s||(s="feature"),e||(e=n.d);var o=t[s]=t[s]||{};(o[r]=o[r]||[]).push([e,i])}},3738:(e,t,r)=>{"use strict";r.d(t,{He:()=>i,Kp:()=>a,Lc:()=>d,Rz:()=>u,TZ:()=>n,bD:()=>s,d3:()=>o,jx:()=>l,sl:()=>f,uP:()=>c});const n=r(860).K7.sessionTrace,i="bstResource",s="resource",o="-start",a="-end",c="fn"+o,d="fn"+a,u="pushState",l=1e3,f=3e4},3785:(e,t,r)=>{"use strict";r.d(t,{R:()=>c,b:()=>d});var n=r(9908),i=r(1863),s=r(860),o=r(3969),a=r(993);function c(e,t,r={},c=a.p_.INFO,d=!0,u,l=(0,i.t)()){(0,n.p)(o.xV,["API/logging/".concat(c.toLowerCase(),"/called")],void 0,s.K7.metrics,e),(0,n.p)(a.ET,[l,t,r,c,d,u],void 0,s.K7.logging,e)}function d(e){return"string"==typeof e&&Object.values(a.p_).some(t=>t===e.toUpperCase().trim())}},3878:(e,t,r)=>{"use strict";function n(e,t){return{capture:e,passive:!1,signal:t}}function i(e,t,r=!1,i){window.addEventListener(e,t,n(r,i))}function s(e,t,r=!1,i){document.addEventListener(e,t,n(r,i))}r.d(t,{DD:()=>s,jT:()=>n,sp:()=>i})},3962:(e,t,r)=>{"use strict";r.d(t,{AM:()=>o,O2:()=>l,OV:()=>s,Qu:()=>f,TZ:()=>c,ih:()=>h,pP:()=>a,t1:()=>u,tC:()=>i,wD:()=>d});var n=r(860);const i=["click","keydown","submit"],s="popstate",o="api",a="initialPageLoad",c=n.K7.softNav,d=5e3,u=500,l={INITIAL_PAGE_LOAD:"",ROUTE_CHANGE:1,UNSPECIFIED:2},f={INTERACTION:1,AJAX:2,CUSTOM_END:3,CUSTOM_TRACER:4},h={IP:"in progress",PF:"pending finish",FIN:"finished",CAN:"cancelled"}},3969:(e,t,r)=>{"use strict";r.d(t,{TZ:()=>n,XG:()=>a,rs:()=>i,xV:()=>o,z_:()=>s});const n=r(860).K7.metrics,i="sm",s="cm",o="storeSupportabilityMetrics",a="storeEventMetrics"},4234:(e,t,r)=>{"use strict";r.d(t,{W:()=>i});var n=r(1687);class i{constructor(e,t){this.agentRef=e,this.ee=e?.ee,this.featureName=t,this.blocked=!1}deregisterDrain(){(0,n.x3)(this.agentRef,this.featureName)}}},4261:(e,t,r)=>{"use strict";r.d(t,{$9:()=>u,BL:()=>c,CH:()=>p,Dl:()=>R,Fw:()=>w,PA:()=>v,Pl:()=>n,Pv:()=>x,Tb:()=>f,U2:()=>o,V1:()=>A,Wb:()=>T,bt:()=>b,cD:()=>y,d3:()=>E,dT:()=>d,eY:()=>g,fF:()=>h,hG:()=>s,hw:()=>i,k6:()=>a,nb:()=>m,o5:()=>l});const n="api-",i=n+"ixn-",s="addPageAction",o="addToTrace",a="addRelease",c="finished",d="interaction",u="log",l="noticeError",f="pauseReplay",h="recordCustomEvent",p="recordReplay",g="register",m="setApplicationVersion",v="setCurrentRouteName",y="setCustomAttribute",b="setErrorHandler",w="setPageViewName",R="setUserId",E="start",T="wrapLogger",A="measure",x="consent"},5205:(e,t,r)=>{"use strict";r.d(t,{j:()=>x});var n=r(384),i=r(1741);var s=r(2555),o=r(3333);const a=e=>{if(!e||"string"!=typeof e)return!1;try{document.createDocumentFragment().querySelector(e)}catch{return!1}return!0};var c=r(2614),d=r(944),u=r(8122);const l="[data-nr-mask]",f=e=>(0,u.a)(e,(()=>{const e={feature_flags:[],experimental:{register:!1,resources:!1},mask_selector:"*",block_selector:"[data-nr-block]",mask_input_options:{color:!1,date:!1,"datetime-local":!1,email:!1,month:!1,number:!1,range:!1,search:!1,tel:!1,text:!1,time:!1,url:!1,week:!1,textarea:!1,select:!1,password:!0}};return{ajax:{deny_list:void 0,block_internal:!0,enabled:!0,autoStart:!0},api:{register:{get enabled(){return e.feature_flags.includes(o.$v.REGISTER)||e.experimental.register},set enabled(t){e.experimental.register=t},duplicate_data_to_container:!1}},browser_consent_mode:{enabled:!1},distributed_tracing:{enabled:void 0,exclude_newrelic_header:void 0,cors_use_newrelic_header:void 0,cors_use_tracecontext_headers:void 0,allowed_origins:void 0},get feature_flags(){return e.feature_flags},set feature_flags(t){e.feature_flags=t},generic_events:{enabled:!0,autoStart:!0},harvest:{interval:30},jserrors:{enabled:!0,autoStart:!0},logging:{enabled:!0,autoStart:!0},metrics:{enabled:!0,autoStart:!0},obfuscate:void 0,page_action:{enabled:!0},page_view_event:{enabled:!0,autoStart:!0},page_view_timing:{enabled:!0,autoStart:!0},performance:{capture_marks:!1,capture_measures:!1,capture_detail:!0,resources:{get enabled(){return e.feature_flags.includes(o.$v.RESOURCES)||e.experimental.resources},set enabled(t){e.experimental.resources=t},asset_types:[],first_party_domains:[],ignore_newrelic:!0}},privacy:{cookies_enabled:!0},proxy:{assets:void 0,beacon:void 0},session:{expiresMs:c.wk,inactiveMs:c.BB},session_replay:{autoStart:!0,enabled:!1,preload:!1,sampling_rate:10,error_sampling_rate:100,collect_fonts:!1,inline_images:!1,fix_stylesheets:!0,mask_all_inputs:!0,get mask_text_selector(){return e.mask_selector},set mask_text_selector(t){a(t)?e.mask_selector="".concat(t,",").concat(l):""===t||null===t?e.mask_selector=l:(0,d.R)(5,t)},get block_class(){return"nr-block"},get ignore_class(){return"nr-ignore"},get mask_text_class(){return"nr-mask"},get block_selector(){return e.block_selector},set block_selector(t){a(t)?e.block_selector+=",".concat(t):""!==t&&(0,d.R)(6,t)},get mask_input_options(){return e.mask_input_options},set mask_input_options(t){t&&"object"==typeof t?e.mask_input_options={...t,password:!0}:(0,d.R)(7,t)}},session_trace:{enabled:!0,autoStart:!0},soft_navigations:{enabled:!0,autoStart:!0},ssl:void 0,user_actions:{enabled:!0,elementAttributes:["id","className","tagName","type"]}}})());var h=r(6154),p=r(9324);let g=0;const m={buildEnv:p.F3,distMethod:p.Xs,version:p.xv,originTime:h.WN},v={consented:!1},y={activatedFeatures:void 0,appMetadata:{},configured:!1,get consented(){return this.session?.state?.consent||v.consented},set consented(e){v.consented=e},customTransaction:void 0,denyList:[],disabled:!1,drainRegistry:new Map,harvester:void 0,isolatedBacklog:!1,isRecording:!1,loaderType:void 0,maxBytes:3e4,obfuscator:void 0,onerror:void 0,ptid:void 0,releaseIds:{},session:void 0,timeKeeper:void 0,registeredEntities:[],jsAttributesMetadata:{bytes:0},get harvestCount(){return++g}},b=e=>{const t=(0,u.a)(e,y),r=Object.keys(m).reduce((e,t)=>(e[t]={value:m[t],writable:!1,configurable:!0,enumerable:!0},e),{});return Object.defineProperties(t,r)},w=e=>{const t=e.startsWith("http");e+="/",r.p=t?e:"https://"+e};var R=r(7836),E=r(3241);const T={accountID:void 0,trustKey:void 0,agentID:void 0,licenseKey:void 0,applicationID:void 0,xpid:void 0},A=e=>(0,u.a)(e,T);function x(e,t={},r,o){let{init:a,info:c,loader_config:d,runtime:u={},exposed:l=!0}=t;if(!c){const e=(0,n.pV)();a=e.init,c=e.info,d=e.loader_config}e.init=f(a||{}),e.loader_config=A(d||{}),c.jsAttributes??={},h.bv&&(c.jsAttributes.isWorker=!0),e.info=(0,s.D)(c);const p=e.init;e.runtime??=b(u),p.proxy.assets&&w(p.proxy.assets),e.runtime.configured||(Object.defineProperty(e,"beacons",{get:()=>[e.info.beacon,e.info.errorBeacon,e.init.proxy.assets,e.init.proxy.beacon].filter(Boolean)}),Object.defineProperty(e.runtime,"denyList",{get:()=>[...e.init.ajax.deny_list||[],...e.init.ajax.block_internal?e.beacons:[]]}),e.runtime.ptid=e.agentIdentifier,function(e){const t=(0,n.pV)();Object.getOwnPropertyNames(i.W.prototype).forEach(r=>{const n=i.W.prototype[r];if("function"!=typeof n||"constructor"===n)return;let s=t[r];e[r]&&!1!==e.exposed&&"micro-agent"!==e.runtime?.loaderType&&(t[r]=(...t)=>{const n=e[r](...t);return s?s(...t):n})})}(e),e.runtime.loaderType=r,e.ee=R.ee.get(e.agentIdentifier),e.exposed=l,(0,E.W)({drained:!!e.runtime.activatedFeatures,type:"lifecycle",name:"initialize",feature:void 0,data:e.config}),e.runtime.configured=!0)}},5270:(e,t,r)=>{"use strict";r.d(t,{Aw:()=>o,SR:()=>s,rF:()=>a});var n=r(384),i=r(7767);function s(e){return!!(0,n.dV)().o.MO&&(0,i.V)(e)&&!0===e?.session_trace.enabled}function o(e){return!0===e?.session_replay.preload&&s(e)}function a(e,t){try{if("string"==typeof t?.type){if("password"===t.type.toLowerCase())return"*".repeat(e?.length||0);if(void 0!==t?.dataset?.nrUnmask||t?.classList?.contains("nr-unmask"))return e}}catch(e){}return"string"==typeof e?e.replace(/[\S]/g,"*"):"*".repeat(e?.length||0)}},5289:(e,t,r)=>{"use strict";r.d(t,{GG:()=>o,Qr:()=>c,sB:()=>a});var n=r(3878),i=r(6389);function s(){return"undefined"==typeof document||"complete"===document.readyState}function o(e,t){if(s())return e();const r=(0,i.J)(e),o=setInterval(()=>{s()&&(clearInterval(o),r())},500);(0,n.sp)("load",r,t)}function a(e){if(s())return e();(0,n.DD)("DOMContentLoaded",e)}function c(e){if(s())return e();(0,n.sp)("popstate",e)}},5607:(e,t,r)=>{"use strict";r.d(t,{W:()=>n});const n=(0,r(9566).bz)()},5732:(e,t,r)=>{"use strict";r.d(t,{$5:()=>u,B5:()=>d,Ms:()=>s,Ux:()=>a,YA:()=>c,fQ:()=>i,yx:()=>o});var n=r(7508);const i={MFE:"MFE",BA:"BA"};function s(e,t){if(!e||!t?.init.api.register.enabled)return[];const r=t.runtime.registeredEntities;return r?.filter(t=>String(t.metadata.target.id)===String(e)).map(e=>e.metadata.target)||[]}function o(e,t){if(!e||!t?.init.api.register.enabled)return[];const r=t.runtime.registeredEntities;return r?.filter(t=>t.metadata.timings?.asset?.endsWith(e)).map(e=>e.metadata.target)||[]}function a(e,t){if(!l(t))return{};const r=t.agentRef.runtime.appMetadata.agents[0].entityGuid;return e?e.attributes:{"entity.guid":r,appId:t.agentRef.info.applicationID}}function c(e,t){return d(e,t)?{"child.id":e.id,"child.type":e.type,...a(void 0,t)}:{}}function d(e,t){return!!e&&!!l(t)&&t.agentRef.init.api.register.duplicate_data_to_container}function u(e){if(!e?.init.api.register.enabled)return[void 0];const t=[];try{var r=(0,n.AZ)((0,n.QL)());let i=r.length-1;for(;r[i];)t.push(...o(r[i--],e))}catch(e){}return t.length||t.push(void 0),t}function l(e){return 2===e?.harvestEndpointVersion}},6154:(e,t,r)=>{"use strict";r.d(t,{OF:()=>d,RI:()=>i,WN:()=>f,bv:()=>s,gm:()=>o,lR:()=>l,m:()=>c,mw:()=>a,sb:()=>u,zk:()=>h});var n=r(1863);const i="undefined"!=typeof window&&!!window.document,s="undefined"!=typeof WorkerGlobalScope&&("undefined"!=typeof self&&self instanceof WorkerGlobalScope&&self.navigator instanceof WorkerNavigator||"undefined"!=typeof globalThis&&globalThis instanceof WorkerGlobalScope&&globalThis.navigator instanceof WorkerNavigator),o=i?window:"undefined"!=typeof WorkerGlobalScope&&("undefined"!=typeof self&&self instanceof WorkerGlobalScope&&self||"undefined"!=typeof globalThis&&globalThis instanceof WorkerGlobalScope&&globalThis),a=Boolean("hidden"===o?.document?.visibilityState),c=""+o?.location,d=/iPad|iPhone|iPod/.test(o.navigator?.userAgent),u=d&&"undefined"==typeof SharedWorker,l=(()=>{const e=o.navigator?.userAgent?.match(/Firefox[/\s](\d+\.\d+)/);return Array.isArray(e)&&e.length>=2?+e[1]:0})(),f=Date.now()-(0,n.t)(),h=()=>{const e=o?.performance?.getEntriesByType?.("navigation")?.[0];if(e&&e.responseStart>0&&e.responseStart<o.performance.now())return e}},6344:(e,t,r)=>{"use strict";r.d(t,{BB:()=>u,Qb:()=>l,TZ:()=>i,Ug:()=>o,Vh:()=>s,_s:()=>a,bc:()=>d,yP:()=>c});var n=r(2614);const i=r(860).K7.sessionReplay,s="errorDuringReplay",o=.12,a={DomContentLoaded:0,Load:1,FullSnapshot:2,IncrementalSnapshot:3,Meta:4,Custom:5},c={[n.g.ERROR]:15e3,[n.g.FULL]:3e5,[n.g.OFF]:0},d={RESET:{message:"Session was reset",sm:"Reset"},IMPORT:{message:"Recorder failed to import",sm:"Import"},TOO_MANY:{message:"429: Too Many Requests",sm:"Too-Many"},TOO_BIG:{message:"Payload was too large",sm:"Too-Big"},CROSS_TAB:{message:"Session Entity was set to OFF on another tab",sm:"Cross-Tab"},ENTITLEMENTS:{message:"Session Replay is not allowed and will not be started",sm:"Entitlement"}},u=5e3,l={API:"api",RESUME:"resume",SWITCH_TO_FULL:"switchToFull",INITIALIZE:"initialize",PRELOAD:"preload"}},6389:(e,t,r)=>{"use strict";function n(e,t=500,r={}){const n=r?.leading||!1;let i;return(...r)=>{n&&void 0===i&&(e.apply(this,r),i=setTimeout(()=>{i=clearTimeout(i)},t)),n||(clearTimeout(i),i=setTimeout(()=>{e.apply(this,r)},t))}}function i(e){let t=!1;return(...r)=>{t||(t=!0,e.apply(this,r))}}r.d(t,{J:()=>i,s:()=>n})},6630:(e,t,r)=>{"use strict";r.d(t,{T:()=>n});const n=r(860).K7.pageViewEvent},6774:(e,t,r)=>{"use strict";r.d(t,{T:()=>n});const n=r(860).K7.jserrors},7295:(e,t,r)=>{"use strict";r.d(t,{Xv:()=>o,gX:()=>i,iW:()=>s});var n=[];function i(e){if(!e||s(e))return!1;if(0===n.length)return!0;if("*"===n[0].hostname)return!1;for(var t=0;t<n.length;t++){var r=n[t];if(r.hostname.test(e.hostname)&&r.pathname.test(e.pathname))return!1}return!0}function s(e){return void 0===e.hostname}function o(e){if(n=[],e&&e.length)for(var t=0;t<e.length;t++){let r=e[t];if(!r)continue;if("*"===r)return void(n=[{hostname:"*"}]);0===r.indexOf("http://")?r=r.substring(7):0===r.indexOf("https://")&&(r=r.substring(8));const i=r.indexOf("/");let s,o;i>0?(s=r.substring(0,i),o=r.substring(i)):(s=r,o="*");let[c]=s.split(":");n.push({hostname:a(c),pathname:a(o,!0)})}}function a(e,t=!1){const r=e.replace(/[.+?^${}()|[\]\\]/g,e=>"\\"+e).replace(/\*/g,".*?");return new RegExp((t?"^":"")+r+"$")}},7485:(e,t,r)=>{"use strict";r.d(t,{D:()=>i});var n=r(6154);function i(e){if(0===(e||"").indexOf("data:"))return{protocol:"data"};try{const t=new URL(e,location.href),r={port:t.port,hostname:t.hostname,pathname:t.pathname,search:t.search,protocol:t.protocol.slice(0,t.protocol.indexOf(":")),sameOrigin:t.protocol===n.gm?.location?.protocol&&t.host===n.gm?.location?.host};return r.port&&""!==r.port||("http:"===t.protocol&&(r.port="80"),"https:"===t.protocol&&(r.port="443")),r.pathname&&""!==r.pathname?r.pathname.startsWith("/")||(r.pathname="/".concat(r.pathname)):r.pathname="/",r}catch(e){return{}}}},7508:(e,t,r)=>{"use strict";r.d(t,{AZ:()=>g,Qr:()=>b,QL:()=>m});var n=r(6154),i=r(1863),s=r(9119),o=r(7866);class a{dom=new c;performance=new c;constructor(e){this.url=e}get script(){const e=Math.max(this.dom.start,this.performance.end);return{start:e,end:Math.max(this.dom.end,this.performance.end,e)}}}class c{start=0;end=0;value=void 0}let d;try{d=g(m())[0]}catch(e){d=g(e)[0]}const u=e=>"script"===e.initiatorType||["link","fetch"].includes(e.initiatorType)&&e.name.endsWith(".js"),l=new Map;let f=[];function h(e){return l.get(e)}function p(e){const t=h(e);if(t)return t;const r=new a(e);if(l.set(e,r),l.size>1e3){const e=l.keys().next().value;l.delete(e)}return r}if(n.gm.MutationObserver&&n.gm.document){new MutationObserver(e=>{e.forEach(e=>{e.addedNodes.forEach(e=>{if("SCRIPT"===e.nodeName&&e.src){const t=p((0,s.L)(e.src));t.dom.start=(0,i.t)(),t.dom.value=e;const r=()=>{t.dom.end=(0,i.t)()};["load","error"].forEach(t=>e.addEventListener(t,r,{once:!0}))}})})}).observe(n.gm.document,{childList:!0,subtree:!0})}if(n.gm.PerformanceObserver?.supportedEntryTypes.includes("resource")){new PerformanceObserver(e=>{e.getEntries().filter(u).forEach(e=>{const t=p((0,s.L)(e.name));t.performance.start=Math.floor(e.startTime),t.performance.end=Math.floor(e.responseEnd),t.performance.value=e;const r=[];f.forEach(({test:t,addedAt:n},s)=>{(t(e)||(0,i.t)()-n>1e4)&&r.push(s)}),f=f.filter((e,t)=>!r.includes(t))})}).observe({type:"resource",buffered:!0})}function g(e){if(!e||"string"!=typeof e)return[];const t=new Set,r=e.split("\n");for(const e of r){const r=e.match(o.cn)||e.match(o.hB)||e.match(o.fL);if(r&&r[2])t.add((0,s.L)(r[2]));else{const r=e.match(/\(([^)]+\.js):\d+:\d+\)/)||e.match(/^\s+at\s+([^\s(]+\.js):\d+:\d+/);r&&r[1]&&t.add((0,s.L)(r[1]))}}return[...t]}function m(){let e;try{const t=Error.stackTraceLimit;Error.stackTraceLimit=50,e=(new Error).stack,Error.stackTraceLimit=t}catch(t){e=(new Error).stack}return e}function v(e,t){return(0,s.L)(e.name)===t}function y(e,t){e.fetchStart=Math.floor(t.startTime),e.fetchEnd=Math.floor(t.responseEnd),e.asset=t.name,e.type=t.initiatorType}function b(){const e={registeredAt:(0,i.t)(),reportedAt:void 0,fetchStart:0,fetchEnd:0,scriptStart:0,scriptEnd:0,asset:void 0,type:"unknown"},t=m();if(!t)return e;const r=n.gm.performance?.getEntriesByType("navigation")?.[0]?.name||"";try{const o=g(t),a=(o.length>1?o.filter(e=>d!==e):o)[0];if(!a)return e;if(r.includes(a))return e.asset=(0,s.L)(r),e.type="inline",e;e.correlation=h(a);const c=e.correlation?.performance.value||performance.getEntriesByType("resource").find(e=>v(e,a));c?y(e,c):function(e){if(!e||!n.gm.document)return!1;try{const t=n.gm.document.querySelectorAll('link[rel="preload"][as="script"]');for(const r of t)if((0,s.L)(r.href)===e)return!0}catch(e){}return!1}(a)&&(e.asset=a,e.type="preload",f.push({addedAt:(0,i.t)(),test:t=>!!v(t,a)&&(y(e,t),!0)})),Object.defineProperty(e,"scriptStart",{get:()=>e.correlation?.script.start||e.fetchEnd}),Object.defineProperty(e,"scriptEnd",{get:()=>e.correlation?.script.end||e.registeredAt})}catch(e){}return e}},7699:(e,t,r)=>{"use strict";r.d(t,{It:()=>s,KC:()=>a,No:()=>i,qh:()=>o});var n=r(860);const i=16e3,s=1e6,o="SESSION_ERROR",a={[n.K7.logging]:!0,[n.K7.genericEvents]:!0,[n.K7.jserrors]:!0,[n.K7.ajax]:!0}},7767:(e,t,r)=>{"use strict";r.d(t,{V:()=>i});var n=r(6154);const i=e=>n.RI&&!0===e?.privacy.cookies_enabled},7836:(e,t,r)=>{"use strict";r.d(t,{P:()=>a,ee:()=>c});var n=r(384),i=r(8990),s=r(2646),o=r(5607);const a="nr@context:".concat(o.W),c=function e(t,r){var n={},o={},u={},l=!1;try{l=16===r.length&&d.initializedAgents?.[r]?.runtime.isolatedBacklog}catch(e){}var f={on:p,addEventListener:p,removeEventListener:function(e,t){var r=n[e];if(!r)return;for(var i=0;i<r.length;i++)r[i]===t&&r.splice(i,1)},emit:function(e,r,n,i,s){!1!==s&&(s=!0);if(c.aborted&&!i)return;t&&s&&t.emit(e,r,n);var a=h(n);g(e).forEach(e=>{e.apply(a,r)});var d=v()[o[e]];d&&d.push([f,e,r,a]);return a},get:m,listeners:g,context:h,buffer:function(e,t){const r=v();if(t=t||"feature",f.aborted)return;Object.entries(e||{}).forEach(([e,n])=>{o[n]=t,t in r||(r[t]=[])})},abort:function(){f._aborted=!0,Object.keys(f.backlog).forEach(e=>{delete f.backlog[e]})},isBuffering:function(e){return!!v()[o[e]]},debugId:r,backlog:l?{}:t&&"object"==typeof t.backlog?t.backlog:{},isolatedBacklog:l};return Object.defineProperty(f,"aborted",{get:()=>{let e=f._aborted||!1;return e||(t&&(e=t.aborted),e)}}),f;function h(e){return e&&e instanceof s.y?e:e?(0,i.I)(e,a,()=>new s.y(a)):new s.y(a)}function p(e,t){n[e]=g(e).concat(t)}function g(e){return n[e]||[]}function m(t){return u[t]=u[t]||e(f,t)}function v(){return f.backlog}}(void 0,"globalEE"),d=(0,n.Zm)();d.ee||(d.ee=c)},7866:(e,t,r)=>{"use strict";r.d(t,{Nc:()=>s,cn:()=>a,fL:()=>i,h3:()=>n,hB:()=>o});const n=/function (.+?)\s*\(/,i=/^\s*at .+ \(eval at \S+ \((?:(?:file|http|https):[^)]+)?\)(?:, [^:]*:\d+:\d+)?\)$/i,s=/^\s*at Function code \(Function code:\d+:\d+\)\s*/i,o=/^\s*at (?:((?:\[object object\])?(?:[^(]*\([^)]*\))*[^()]*(?: \[as \S+\])?) )?\(?((?:file|http|https|chrome-extension):.*?)?:(\d+)(?::(\d+))?\)?\s*$/i,a=/^\s*(?:([^@]*)(?:\(.*?\))?@)?((?:file|http|https|chrome|safari-extension).*?):(\d+)(?::(\d+))?\s*$/i},8122:(e,t,r)=>{"use strict";r.d(t,{a:()=>i});var n=r(944);function i(e,t){try{if(!e||"object"!=typeof e)return(0,n.R)(3);if(!t||"object"!=typeof t)return(0,n.R)(4);const r=Object.create(Object.getPrototypeOf(t),Object.getOwnPropertyDescriptors(t)),s=0===Object.keys(r).length?e:r;for(let o in s)if(void 0!==e[o])try{if(null===e[o]){r[o]=null;continue}Array.isArray(e[o])&&Array.isArray(t[o])?r[o]=Array.from(new Set([...e[o],...t[o]])):e[o]instanceof Map||e[o]instanceof Set||e[o]instanceof Date||e[o]instanceof RegExp?r[o]=e[o]:"object"==typeof e[o]&&"object"==typeof t[o]?r[o]=i(e[o],t[o]):r[o]=e[o]}catch(e){r[o]||(0,n.R)(1,e)}return r}catch(e){(0,n.R)(2,e)}}},8139:(e,t,r)=>{"use strict";r.d(t,{u:()=>f});var n=r(7836),i=r(3434),s=r(8990),o=r(6154);const a={},c=o.gm.XMLHttpRequest,d="addEventListener",u="removeEventListener",l="nr@wrapped:".concat(n.P);function f(e){var t=function(e){return(e||n.ee).get("events")}(e);if(a[t.debugId]++)return t;a[t.debugId]=1;var r=(0,i.YM)(t,!0);function f(e){r.inPlace(e,[d,u],"-",p)}function p(e,t){return e[1]}return"getPrototypeOf"in Object&&(o.RI&&h(document,f),c&&h(c.prototype,f),h(o.gm,f)),t.on(d+"-start",function(e,t){var n=e[1];if(null!==n&&("function"==typeof n||"object"==typeof n)&&"newrelic"!==e[0]){var i=(0,s.I)(n,l,function(){var e={object:function(){if("function"!=typeof n.handleEvent)return;return n.handleEvent.apply(n,arguments)},function:n}[typeof n];return e?r(e,"fn-",null,e.name||"anonymous"):n});this.wrapped=e[1]=i}}),t.on(u+"-start",function(e){e[1]=this.wrapped||e[1]}),t}function h(e,t,...r){let n=e;for(;"object"==typeof n&&!Object.prototype.hasOwnProperty.call(n,d);)n=Object.getPrototypeOf(n);n&&t(n,...r)}},8362:(e,t,r)=>{"use strict";r.d(t,{d:()=>s});var n=r(9566),i=r(1741);class s extends i.W{agentIdentifier=(0,n.LA)(16)}},8374:(e,t,r)=>{r.nc=(()=>{try{return document?.currentScript?.nonce}catch(e){}return""})()},8990:(e,t,r)=>{"use strict";r.d(t,{I:()=>i});var n=Object.prototype.hasOwnProperty;function i(e,t,r){if(n.call(e,t))return e[t];var i=r();if(Object.defineProperty&&Object.keys)try{return Object.defineProperty(e,t,{value:i,writable:!0,enumerable:!1}),i}catch(e){}return e[t]=i,i}},9119:(e,t,r)=>{"use strict";r.d(t,{L:()=>s});var n=/([^?#]*)[^#]*(#[^?]*|$).*/,i=/([^?#]*)().*/;function s(e,t){return e?e.replace(t?n:i,"$1$2"):e}},9300:(e,t,r)=>{"use strict";r.d(t,{T:()=>n,f:()=>i});const n=r(860).K7.ajax,i="ajaxRequest.id"},9324:(e,t,r)=>{"use strict";r.d(t,{AJ:()=>o,F3:()=>i,Xs:()=>s,Yq:()=>a,xv:()=>n});const n="1.314.0",i="PROD",s="CDN",o="@newrelic/rrweb",a="1.1.0"},9566:(e,t,r)=>{"use strict";r.d(t,{LA:()=>a,ZF:()=>c,bz:()=>o,el:()=>d});var n=r(6154);const i="xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx";function s(e,t){return e?15&e[t]:16*Math.random()|0}function o(){const e=n.gm?.crypto||n.gm?.msCrypto;let t,r=0;return e&&e.getRandomValues&&(t=e.getRandomValues(new Uint8Array(30))),i.split("").map(e=>"x"===e?s(t,r++).toString(16):"y"===e?(3&s()|8).toString(16):e).join("")}function a(e){const t=n.gm?.crypto||n.gm?.msCrypto;let r,i=0;t&&t.getRandomValues&&(r=t.getRandomValues(new Uint8Array(e)));const o=[];for(var a=0;a<e;a++)o.push(s(r,i++).toString(16));return o.join("")}function c(){return a(16)}function d(){return a(32)}},9908:(e,t,r)=>{"use strict";r.d(t,{d:()=>n,p:()=>i});var n=r(7836).ee.get("handle");function i(e,t,r,i,s){s?(s.buffer([e],i),s.emit(e,t,r)):(n.buffer([e],i),n.emit(e,t,r))}}},n={};function i(e){var t=n[e];if(void 0!==t)return t.exports;var s=n[e]={exports:{}};return r[e](s,s.exports,i),s.exports}i.m=r,i.d=(e,t)=>{for(var r in t)i.o(t,r)&&!i.o(e,r)&&Object.defineProperty(e,r,{enumerable:!0,get:t[r]})},i.f={},i.e=e=>Promise.all(Object.keys(i.f).reduce((t,r)=>(i.f[r](e,t),t),[])),i.u=e=>({212:"nr-spa-compressor",249:"nr-spa-recorder",478:"nr-spa"}[e]+"-1.314.0.min.js"),i.o=(e,t)=>Object.prototype.hasOwnProperty.call(e,t),e={},t="NRBA-1.314.0.PROD:",i.l=(r,n,s,o)=>{if(e[r])e[r].push(n);else{var a,c;if(void 0!==s)for(var d=document.getElementsByTagName("script"),u=0;u<d.length;u++){var l=d[u];if(l.getAttribute("src")==r||l.getAttribute("data-webpack")==t+s){a=l;break}}if(!a){c=!0;var f={478:"sha512-mgj9qD5BdNJpEJCPmd1OfPzfnL/HRJ64OPYq/GZQxKc0756ytnQixcg8IA8iZWe/PRWN4DuGAnX82J2lscbiIw==",249:"sha512-fUJXIKjSt6UL9n1UZzNMzGGL3CxnBT/AgtoTat0/ONINbEB2PoiO+yK54DPYDwzpXDBddJHvs2BrYFj1GBpo2Q==",212:"sha512-vjoeBTgR0ONx6WcgBdfuPgcta1834zMPDhLL05C9c/pB+/xAuxL4BJMu9yeXYh+oIV7oIn6qTkDC7rBcZIDiHg=="};(a=document.createElement("script")).charset="utf-8",i.nc&&a.setAttribute("nonce",i.nc),a.setAttribute("data-webpack",t+s),a.src=r,0!==a.src.indexOf(window.location.origin+"/")&&(a.crossOrigin="anonymous"),f[o]&&(a.integrity=f[o])}e[r]=[n];var h=(t,n)=>{a.onerror=a.onload=null,clearTimeout(p);var i=e[r];if(delete e[r],a.parentNode&&a.parentNode.removeChild(a),i&&i.forEach(e=>e(n)),t)return t(n)},p=setTimeout(h.bind(null,void 0,{type:"timeout",target:a}),12e4);a.onerror=h.bind(null,a.onerror),a.onload=h.bind(null,a.onload),c&&document.head.appendChild(a)}},i.r=e=>{"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},i.p="https://js-agent.newrelic.com/",(()=>{var e={38:0,788:0};i.f.j=(t,r)=>{var n=i.o(e,t)?e[t]:void 0;if(0!==n)if(n)r.push(n[2]);else{var s=new Promise((r,i)=>n=e[t]=[r,i]);r.push(n[2]=s);var o=i.p+i.u(t),a=new Error;i.l(o,r=>{if(i.o(e,t)&&(0!==(n=e[t])&&(e[t]=void 0),n)){var s=r&&("load"===r.type?"missing":r.type),o=r&&r.target&&r.target.src;a.message="Loading chunk "+t+" failed: ("+s+": "+o+")",a.name="ChunkLoadError",a.type=s,a.request=o,n[1](a)}},"chunk-"+t,t)}};var t=(t,r)=>{var n,s,[o,a,c]=r,d=0;if(o.some(t=>0!==e[t])){for(n in a)i.o(a,n)&&(i.m[n]=a[n]);if(c)c(i)}for(t&&t(r);d<o.length;d++)s=o[d],i.o(e,s)&&e[s]&&e[s][0](),e[s]=0},r=self["webpackChunk:NRBA-1.314.0.PROD"]=self["webpackChunk:NRBA-1.314.0.PROD"]||[];r.forEach(t.bind(null,0)),r.push=t.bind(null,r.push.bind(r))})(),(()=>{"use strict";i(8374);var e=i(8362),t=i(860);const r=Object.values(t.K7);var n=i(5205);var s=i(9908),o=i(1863),a=i(4261),c=i(1738);var d=i(1687),u=i(4234),l=i(5289),f=i(6154),h=i(944),p=i(5270),g=i(7767),m=i(6389),v=i(7699);class y extends u.W{constructor(e,t){super(e,t),this.abortHandler=void 0,this.featAggregate=void 0,this.loadedSuccessfully=void 0,this.onAggregateImported=new Promise(e=>{this.loadedSuccessfully=e}),this.deferred=Promise.resolve(),!1===e.init[this.featureName].autoStart?this.deferred=new Promise((t,r)=>{this.ee.on("manual-start-all",(0,m.J)(()=>{(0,d.Ak)(e,this.featureName),t()}))}):(0,d.Ak)(e,t)}importAggregator(e,t,r={}){if(this.featAggregate)return;const n=async()=>{let n;await this.deferred;try{if((0,g.V)(e.init)){const{setupAgentSession:t}=await i.e(478).then(i.bind(i,8766));n=t(e)}}catch(e){(0,h.R)(20,e),this.ee.emit("internal-error",[e]),(0,s.p)(v.qh,[e],void 0,this.featureName,this.ee)}try{if(!this.#t(this.featureName,n,e.init))return(0,d.Ze)(this.agentRef,this.featureName),void this.loadedSuccessfully(!1);const{Aggregate:i}=await t();this.featAggregate=new i(e,r),e.runtime.harvester.initializedAggregates.push(this.featAggregate),this.loadedSuccessfully(!0)}catch(e){(0,h.R)(34,e),this.abortHandler?.(),(0,d.Ze)(this.agentRef,this.featureName,!0),this.loadedSuccessfully(!1),this.ee&&this.ee.abort()}};f.RI?(0,l.GG)(()=>n(),!0):n()}#t(e,r,n){if(this.blocked)return!1;switch(e){case t.K7.sessionReplay:return(0,p.SR)(n)&&!!r;case t.K7.sessionTrace:return!!r;default:return!0}}}var b=i(6630),w=i(2614),R=i(3241);class E extends y{static featureName=b.T;constructor(e){var t;super(e,b.T),this.setupInspectionEvents(),t=e,(0,c.Y)(a.Fw,function(e,r){"string"==typeof e&&("/"!==e.charAt(0)&&(e="/"+e),t.runtime.customTransaction=(r||"http://custom.transaction")+e,(0,s.p)(a.Pl+a.Fw,[(0,o.t)()],void 0,void 0,t.ee))},t),this.importAggregator(e,()=>i.e(478).then(i.bind(i,5839)))}setupInspectionEvents(){const e=(e,t)=>{e&&(0,R.W)({timeStamp:e.timeStamp,loaded:"complete"===e.target.readyState,type:"window",name:t,data:e.target.location+""})};(0,l.sB)(t=>{e(t,"DOMContentLoaded")}),(0,l.GG)(t=>{e(t,"load")}),(0,l.Qr)(t=>{e(t,"navigate")}),this.ee.on(w.tS.UPDATE,(e,t)=>{(0,R.W)({type:"lifecycle",name:"session",data:t})})}}var T=i(384);class A extends e.d{constructor(e){var t;(super(),f.gm)?(this.features={},(0,T.bQ)(this.agentIdentifier,this),this.desiredFeatures=new Set(e.features||[]),this.desiredFeatures.add(E),(0,n.j)(this,e,e.loaderType||"agent"),t=this,(0,c.Y)(a.cD,function(e,r,n=!1){if("string"==typeof e){if(["string","number","boolean"].includes(typeof r)||null===r)return(0,c.U)(t,e,r,a.cD,n);(0,h.R)(40,typeof r)}else(0,h.R)(39,typeof e)},t),function(e){(0,c.Y)(a.Dl,function(t,r=!1){if("string"!=typeof t&&null!==t)return void(0,h.R)(41,typeof t);const n=e.info.jsAttributes["enduser.id"];r&&null!=n&&n!==t?(0,s.p)(a.Pl+"setUserIdAndResetSession",[t],void 0,"session",e.ee):(0,c.U)(e,"enduser.id",t,a.Dl,!0)},e)}(this),function(e){(0,c.Y)(a.nb,function(t){if("string"==typeof t||null===t)return(0,c.U)(e,"application.version",t,a.nb,!1);(0,h.R)(42,typeof t)},e)}(this),function(e){(0,c.Y)(a.d3,function(){e.ee.emit("manual-start-all")},e)}(this),function(e){(0,c.Y)(a.Pv,function(t=!0){if("boolean"==typeof t){if((0,s.p)(a.Pl+a.Pv,[t],void 0,"session",e.ee),e.runtime.consented=t,t){const t=e.features.page_view_event;t.onAggregateImported.then(e=>{const r=t.featAggregate;e&&!r.sentRum&&r.sendRum()})}}else(0,h.R)(65,typeof t)},e)}(this),this.run()):(0,h.R)(21)}get config(){return{info:this.info,init:this.init,loader_config:this.loader_config,runtime:this.runtime}}get api(){return this}run(){try{const e=function(e){const t={};return r.forEach(r=>{t[r]=!!e[r]?.enabled}),t}(this.init),n=[...this.desiredFeatures];n.sort((e,r)=>t.P3[e.featureName]-t.P3[r.featureName]),n.forEach(r=>{if(!e[r.featureName]&&r.featureName!==t.K7.pageViewEvent)return;const n=function(e){switch(e){case t.K7.ajax:return[t.K7.jserrors];case t.K7.sessionTrace:return[t.K7.ajax,t.K7.pageViewEvent];case t.K7.sessionReplay:return[t.K7.sessionTrace];case t.K7.pageViewTiming:return[t.K7.pageViewEvent];default:return[]}}(r.featureName).filter(e=>!(e in this.features));n.length>0&&(0,h.R)(36,{targetFeature:r.featureName,missingDependencies:n}),this.features[r.featureName]=new r(this)})}catch(e){(0,h.R)(22,e);for(const e in this.features)this.features[e].abortHandler?.();const t=(0,T.Zm)();delete t.initializedAgents[this.agentIdentifier]?.features,delete this.sharedAggregator;return t.ee.get(this.agentIdentifier).abort(),!1}}}var x=i(2843),S=i(782);class _ extends y{static featureName=S.T;constructor(e){super(e,S.T),f.RI&&((0,x.u)(()=>(0,s.p)("docHidden",[(0,o.t)()],void 0,S.T,this.ee),!0),(0,x.G)(()=>(0,s.p)("winPagehide",[(0,o.t)()],void 0,S.T,this.ee)),this.importAggregator(e,()=>i.e(478).then(i.bind(i,9917))))}}var O=i(3969);class P extends y{static featureName=O.TZ;constructor(e){super(e,O.TZ),this.importAggregator(e,()=>i.e(478).then(i.bind(i,6555)))}}var k=i(6774),N=i(3878),j=i(3304);class D{constructor(e,t,r,n,i){this.name="UncaughtError",this.message="string"==typeof e?e:(0,j.A)(e),this.sourceURL=t,this.line=r,this.column=n,this.__newrelic=i}}function C(e){return M(e)?e:new D(void 0!==e?.message?e.message:e,e?.filename||e?.sourceURL,e?.lineno||e?.line,e?.colno||e?.col,e?.__newrelic,e?.cause)}function L(e){const t="Unhandled Promise Rejection: ";if(!e?.reason)return;if(M(e.reason)){try{e.reason.message.startsWith(t)||(e.reason.message=t+e.reason.message)}catch(e){}return C(e.reason)}const r=C(e.reason);return(r.message||"").startsWith(t)||(r.message=t+r.message),r}function I(e){if(e.error instanceof SyntaxError&&!/:\d+$/.test(e.error.stack?.trim())){const t=new D(e.message,e.filename,e.lineno,e.colno,e.error.__newrelic,e.cause);return t.name=SyntaxError.name,t}return M(e.error)?e.error:C(e)}function M(e){return e instanceof Error&&!!e.stack}function B(e,r,n,i,a=(0,o.t)()){"string"==typeof e&&(e=new Error(e)),(0,s.p)("err",[e,a,!1,r,n.runtime.isRecording,void 0,i],void 0,t.K7.jserrors,n.ee),(0,s.p)("uaErr",[],void 0,t.K7.genericEvents,n.ee)}var H=i(5732),K=i(993),W=i(3785);function F(e,{customAttributes:t={},level:r=K.p_.INFO}={},n,i,s=(0,o.t)()){(0,W.R)(n.ee,e,t,r,!1,i,s)}function U(e,r,n,i,c=(0,o.t)()){(0,s.p)(a.Pl+a.hG,[c,e,r,i],void 0,t.K7.genericEvents,n.ee)}function V(e,r,n,i,c=(0,o.t)()){const{start:d,end:u,customAttributes:l}=r||{},f={customAttributes:l||{}};if("object"!=typeof f.customAttributes||"string"!=typeof e||0===e.length)return void(0,h.R)(57);const p=(e,t)=>null==e?t:"number"==typeof e?e:e instanceof PerformanceMark?e.startTime:Number.NaN;if(f.start=p(d,0),f.end=p(u,c),Number.isNaN(f.start)||Number.isNaN(f.end))(0,h.R)(57);else{if(f.duration=f.end-f.start,!(f.duration<0))return(0,s.p)(a.Pl+a.V1,[f,e,i],void 0,t.K7.genericEvents,n.ee),f;(0,h.R)(58)}}function z(e,r={},n,i,c=(0,o.t)()){(0,s.p)(a.Pl+a.fF,[c,e,r,i],void 0,t.K7.genericEvents,n.ee)}var G=i(7508),Y=i(9566);const Z=["name","id","type"],q=new Map([[U,"addPageAction"],[F,"log"],[V,"measure"],[B,"noticeError"],[z,"recordCustomEvent"]]),X={experimental:(0,m.J)(()=>(0,h.R)(54,"newrelic.register")),disabled:(0,m.J)(()=>(0,h.R)(55)),invalidTarget:(0,m.J)(e=>(0,h.R)(48,e)),deregistered:(0,m.J)(()=>(0,h.R)(68))};function Q(e){(0,c.Y)(a.eY,function(t){return J(e,t)},e)}function J(e,r){X.experimental(),r||={},r.instance=(0,Y.LA)(8),r.type=H.fQ.MFE,r.licenseKey||=e.info.licenseKey,r.blocked=!1,("object"!=typeof r.tags||null===r.tags||Array.isArray(r.tags))&&(r.tags={}),r.parent??={get id(){return e.runtime.appMetadata.agents[0].entityGuid},type:H.fQ.BA};const n=(0,G.Qr)(),i={};Object.prototype.hasOwnProperty.call(r,"attributes")||Object.defineProperty(r,"attributes",{get:()=>({...i,"source.id":r.id,"source.name":r.name,"source.type":r.type,"parent.type":r.parent?.type||H.fQ.BA,"parent.id":r.parent?.id})}),Object.entries(r.tags).forEach(([e,t])=>{Z.includes(e)||(i["source.".concat(e)]=t)});let a=()=>{};const c=e.runtime.registeredEntities,d=e=>{r.blocked=!0,a=e};function u(e){return"string"==typeof e&&!!e.trim()&&e.trim().length<501}e.init.api.register.enabled||d(X.disabled),u(r.id)&&u(r.name)||d(()=>X.invalidTarget(r));const l={addPageAction:(t,n={})=>m(U,[t,{...i,...n},e],r),deregister:()=>{p(),d(X.deregistered)},log:(t,n={})=>m(F,[t,{...n,customAttributes:{...i,...n.customAttributes||{}}},e],r),measure:(t,n={})=>m(V,[t,{...n,customAttributes:{...i,...n.customAttributes||{}}},e],r),noticeError:(t,n={})=>m(B,[t,{...i,...n},e],r),recordCustomEvent:(t,n={})=>m(z,[t,{...i,...n},e],r),setApplicationVersion:e=>g("application.version",e),setCustomAttribute:(e,t)=>g(e,t),setUserId:e=>g("enduser.id",e),metadata:{get customAttributes(){return i},target:r,timings:n}},f=()=>(r.blocked&&a(),r.blocked);function p(){if(n.reportedAt)return;n.reportedAt=(0,o.t)();const e=n.fetchEnd-n.fetchStart,t=n.scriptEnd-n.scriptStart;l.recordCustomEvent("MicroFrontEndTiming",{assetUrl:n.asset,assetType:n.type,timeAlive:n.reportedAt-n.registeredAt,timeToBeRequested:n.fetchStart,timeToExecute:t,timeToFetch:e,timeToLoad:e+t,timeToRegister:n.registeredAt})}f()||(c.push(l),(0,x.G)(p));const g=(e,t)=>{f()||(i[e]=t)},m=(r,n,i)=>{if(f()&&r!==J)return;const a=(0,o.t)(),c=q.get(r)||"unknown";(0,s.p)(O.xV,["API/register/".concat(c,"/called")],void 0,t.K7.metrics,e.ee);try{return r(...n,i,a)}catch(e){(0,h.R)(50,e)}};return l}class ee extends y{static featureName=k.T;constructor(e){var t;super(e,k.T),t=e,(0,c.Y)(a.o5,(e,r)=>B(e,r,t),t),function(e){(0,c.Y)(a.bt,function(t){e.runtime.onerror=t},e)}(e),function(e){let t=0;(0,c.Y)(a.k6,function(e,r){++t>10||(this.runtime.releaseIds[e.slice(-200)]=(""+r).slice(-200))},e)}(e),Q(e);try{this.removeOnAbort=new AbortController}catch(e){}this.ee.on("internal-error",(t,r)=>{this.abortHandler&&(0,s.p)("ierr",[C(t),(0,o.t)(),!0,{},e.runtime.isRecording,r],void 0,this.featureName,this.ee)}),f.gm.addEventListener("unhandledrejection",t=>{this.abortHandler&&(0,s.p)("err",[L(t),(0,o.t)(),!1,{unhandledPromiseRejection:1},e.runtime.isRecording],void 0,this.featureName,this.ee)},(0,N.jT)(!1,this.removeOnAbort?.signal)),f.gm.addEventListener("error",t=>{this.abortHandler&&(0,s.p)("err",[I(t),(0,o.t)(),!1,{},e.runtime.isRecording],void 0,this.featureName,this.ee)},(0,N.jT)(!1,this.removeOnAbort?.signal)),this.abortHandler=this.#r,this.importAggregator(e,()=>i.e(478).then(i.bind(i,9377)))}#r(){this.removeOnAbort?.abort(),this.abortHandler=void 0}}var te=i(8990);let re=1;function ne(e){const t=typeof e;return!e||"object"!==t&&"function"!==t?-1:e===f.gm?0:(0,te.I)(e,"nr@id",function(){return re++})}function ie(e){if("string"==typeof e&&e.length)return e.length;if("object"==typeof e){if("undefined"!=typeof ArrayBuffer&&e instanceof ArrayBuffer&&e.byteLength)return e.byteLength;if("undefined"!=typeof Blob&&e instanceof Blob&&e.size)return e.size;if(!("undefined"!=typeof FormData&&e instanceof FormData))try{return(0,j.A)(e).length}catch(e){return}}}var se=i(8139),oe=i(7836),ae=i(3434);const ce={},de=["open","send"];function ue(e,t){var r=e||oe.ee;const n=function(e){return(e||oe.ee).get("xhr")}(r);if(void 0===f.gm.XMLHttpRequest)return n;if(ce[n.debugId]++)return n;ce[n.debugId]=1,(0,se.u)(r);var i=(0,ae.YM)(n),s=f.gm.XMLHttpRequest,o=f.gm.MutationObserver,a=f.gm.Promise,c=f.gm.setInterval,d="readystatechange",u=["onload","onerror","onabort","onloadstart","onloadend","onprogress","ontimeout"],l=[],p=f.gm.XMLHttpRequest=function(e){const r=new s(e),o=n.context(r);o.targets=(0,H.$5)(t);try{n.emit("new-xhr",[r],o),r.addEventListener(d,(a=o,function(){var e=this;e.readyState>3&&!a.resolved&&(a.resolved=!0,n.emit("xhr-resolved",[],e)),i.inPlace(e,u,"fn-",w)}),(0,N.jT)(!1))}catch(e){(0,h.R)(15,e);try{n.emit("internal-error",[e])}catch(e){}}var a;return r};function g(e,t){i.inPlace(t,["onreadystatechange"],"fn-",w)}if(function(e,t){for(var r in e)t[r]=e[r]}(s,p),p.prototype=s.prototype,i.inPlace(p.prototype,de,"-xhr-",w),n.on("send-xhr-start",function(e,t){g(e,t),function(e){l.push(e),o&&(m?m.then(b):c?c(b):(v=-v,y.data=v))}(t)}),n.on("open-xhr-start",g),o){var m=a&&a.resolve();if(!c&&!a){var v=1,y=document.createTextNode(v);new o(b).observe(y,{characterData:!0})}}else r.on("fn-end",function(e){e[0]&&e[0].type===d||b()});function b(){for(var e=0;e<l.length;e++)g(0,l[e]);l.length&&(l=[])}function w(e,t){return t}return n}var le="fetch-",fe=le+"body-",he=["arrayBuffer","blob","json","text","formData"],pe=f.gm.Request,ge=f.gm.Response,me="prototype";const ve={};function ye(e,t){const r=function(e){return(e||oe.ee).get("fetch")}(e);if(!(pe&&ge&&f.gm.fetch))return r;if(ve[r.debugId]++)return r;function n(e,n,i){var s=e[n];"function"==typeof s&&(e[n]=function(){var e=[...arguments];const n={},o=(0,H.$5)(t);var a;r.emit(i+"before-start",[e],n),n[oe.P]&&n[oe.P].dt&&(a=n[oe.P].dt);var c=s.apply(this,e);return r.emit(i+"start",[e,a],c),c.then(function(e){return r.emit(i+"end",[null,e,o],c),e},function(e){throw r.emit(i+"end",[e,void 0,o],c),e})})}return ve[r.debugId]=1,he.forEach(e=>{n(pe[me],e,fe),n(ge[me],e,fe)}),n(f.gm,"fetch",le),r.on(le+"end",function(e,t,n){var i=this;if(i.targets=n||[void 0],t){var s=t.headers.get("content-length");null!==s&&(i.rxSize=s),r.emit(le+"done",[null,t],i)}else r.emit(le+"done",[e],i)}),r}var be=i(7485);class we{constructor(e){this.agentRef=e}generateTracePayload(e){const t=this.agentRef.loader_config;if(!this.shouldGenerateTrace(e)||!t)return null;var r=(t.accountID||"").toString()||null,n=(t.agentID||"").toString()||null,i=(t.trustKey||"").toString()||null;if(!r||!n)return null;var s=(0,Y.ZF)(),o=(0,Y.el)(),a=Date.now(),c={spanId:s,traceId:o,timestamp:a};return(e.sameOrigin||this.isAllowedOrigin(e)&&this.useTraceContextHeadersForCors())&&(c.traceContextParentHeader=this.generateTraceContextParentHeader(s,o),c.traceContextStateHeader=this.generateTraceContextStateHeader(s,a,r,n,i)),(e.sameOrigin&&!this.excludeNewrelicHeader()||!e.sameOrigin&&this.isAllowedOrigin(e)&&this.useNewrelicHeaderForCors())&&(c.newrelicHeader=this.generateTraceHeader(s,o,a,r,n,i)),c}generateTraceContextParentHeader(e,t){return"00-"+t+"-"+e+"-01"}generateTraceContextStateHeader(e,t,r,n,i){return i+"@nr=0-1-"+r+"-"+n+"-"+e+"----"+t}generateTraceHeader(e,t,r,n,i,s){if(!("function"==typeof f.gm?.btoa))return null;var o={v:[0,1],d:{ty:"Browser",ac:n,ap:i,id:e,tr:t,ti:r}};return s&&n!==s&&(o.d.tk=s),btoa((0,j.A)(o))}shouldGenerateTrace(e){return this.agentRef.init?.distributed_tracing?.enabled&&this.isAllowedOrigin(e)}isAllowedOrigin(e){var t=!1;const r=this.agentRef.init?.distributed_tracing;if(e.sameOrigin)t=!0;else if(r?.allowed_origins instanceof Array)for(var n=0;n<r.allowed_origins.length;n++){var i=(0,be.D)(r.allowed_origins[n]);if(e.hostname===i.hostname&&e.protocol===i.protocol&&e.port===i.port){t=!0;break}}return t}excludeNewrelicHeader(){var e=this.agentRef.init?.distributed_tracing;return!!e&&!!e.exclude_newrelic_header}useNewrelicHeaderForCors(){var e=this.agentRef.init?.distributed_tracing;return!!e&&!1!==e.cors_use_newrelic_header}useTraceContextHeadersForCors(){var e=this.agentRef.init?.distributed_tracing;return!!e&&!!e.cors_use_tracecontext_headers}}var Re=i(9300),Ee=i(7295);function Te(e){return"string"==typeof e?e:e instanceof(0,T.dV)().o.REQ?e.url:f.gm?.URL&&e instanceof URL?e.href:void 0}var Ae=["load","error","abort","timeout"],xe=Ae.length,Se=(0,T.dV)().o.REQ,_e=(0,T.dV)().o.XHR;const Oe="X-NewRelic-App-Data";class Pe extends y{static featureName=Re.T;constructor(e){super(e,Re.T),this.dt=new we(e),this.handler=(e,t,r,n)=>(0,s.p)(e,t,r,n,this.ee);try{const e={xmlhttprequest:"xhr",fetch:"fetch",beacon:"beacon"};f.gm?.performance?.getEntriesByType("resource").forEach(r=>{if(r.initiatorType in e&&0!==r.responseStatus){const n={status:r.responseStatus},i={rxSize:r.transferSize,duration:Math.floor(r.duration),cbTime:0};ke(n,r.name),this.handler("xhr",[n,i,r.startTime,r.responseEnd,e[r.initiatorType]],void 0,t.K7.ajax)}})}catch(e){}ye(this.ee,e),ue(this.ee,e),function(e,r,n,i){function a(e){var t=this;t.totalCbs=0,t.called=0,t.cbTime=0,t.end=T,t.ended=!1,t.xhrGuids={},t.lastSize=null,t.loadCaptureCalled=!1,t.params=this.params||{},t.metrics=this.metrics||{},t.latestLongtaskEnd=0,e.addEventListener("load",function(r){x(t,e)},(0,N.jT)(!1)),f.lR||e.addEventListener("progress",function(e){t.lastSize=e.loaded},(0,N.jT)(!1))}function c(e){this.params={method:e[0]},ke(this,e[1]),this.metrics={}}function d(t,r){e.loader_config.xpid&&this.sameOrigin&&r.setRequestHeader("X-NewRelic-ID",e.loader_config.xpid);var n=i.generateTracePayload(this.parsedOrigin);if(n){var s=!1;n.newrelicHeader&&(r.setRequestHeader("newrelic",n.newrelicHeader),s=!0),n.traceContextParentHeader&&(r.setRequestHeader("traceparent",n.traceContextParentHeader),n.traceContextStateHeader&&r.setRequestHeader("tracestate",n.traceContextStateHeader),s=!0),s&&(this.dt=n)}}function u(e,t){var n=this.metrics,i=e[0],s=this;if(n&&i){var a=ie(i);a&&(n.txSize=a)}this.startTime=(0,o.t)(),this.body=i,this.listener=function(e){try{"abort"!==e.type||s.loadCaptureCalled||(s.params.aborted=!0),("load"!==e.type||s.called===s.totalCbs&&(s.onloadCalled||"function"!=typeof t.onload)&&"function"==typeof s.end)&&s.end(t)}catch(e){try{r.emit("internal-error",[e])}catch(e){}}};for(var c=0;c<xe;c++)t.addEventListener(Ae[c],this.listener,(0,N.jT)(!1))}function l(e,t,r){this.cbTime+=e,t?this.onloadCalled=!0:this.called+=1,this.called!==this.totalCbs||!this.onloadCalled&&"function"==typeof r.onload||"function"!=typeof this.end||this.end(r)}function h(e,t){var r=""+ne(e)+!!t;this.xhrGuids&&!this.xhrGuids[r]&&(this.xhrGuids[r]=!0,this.totalCbs+=1)}function p(e,t){var r=""+ne(e)+!!t;this.xhrGuids&&this.xhrGuids[r]&&(delete this.xhrGuids[r],this.totalCbs-=1)}function g(){this.endTime=(0,o.t)()}function m(e,t){t instanceof _e&&"load"===e[0]&&r.emit("xhr-load-added",[e[1],e[2]],t)}function v(e,t){t instanceof _e&&"load"===e[0]&&r.emit("xhr-load-removed",[e[1],e[2]],t)}function y(e,t,r){t instanceof _e&&("onload"===r&&(this.onload=!0),("load"===(e[0]&&e[0].type)||this.onload)&&(this.xhrCbStart=(0,o.t)()))}function b(e,t){this.xhrCbStart&&r.emit("xhr-cb-time",[(0,o.t)()-this.xhrCbStart,this.onload,t],t)}function w(e){var t,r=e[1]||{};if("string"==typeof e[0]?0===(t=e[0]).length&&f.RI&&(t=""+f.gm.location.href):e[0]&&e[0].url?t=e[0].url:f.gm?.URL&&e[0]&&e[0]instanceof URL?t=e[0].href:"function"==typeof e[0].toString&&(t=e[0].toString()),"string"==typeof t&&0!==t.length){t&&(this.parsedOrigin=(0,be.D)(t),this.sameOrigin=this.parsedOrigin.sameOrigin);var n=i.generateTracePayload(this.parsedOrigin);if(n&&(n.newrelicHeader||n.traceContextParentHeader))if(e[0]&&e[0].headers)a(e[0].headers,n)&&(this.dt=n);else{var s={};for(var o in r)s[o]=r[o];s.headers=new Headers(r.headers||{}),a(s.headers,n)&&(this.dt=n),e.length>1?e[1]=s:e.push(s)}}function a(e,t){var r=!1;return t.newrelicHeader&&(e.set("newrelic",t.newrelicHeader),r=!0),t.traceContextParentHeader&&(e.set("traceparent",t.traceContextParentHeader),t.traceContextStateHeader&&e.set("tracestate",t.traceContextStateHeader),r=!0),r}}function R(e,t){this.params={},this.metrics={},this.startTime=(0,o.t)(),this.dt=t;let[r,n={}]=e;ke(this,Te(r));const i=(""+(r&&r instanceof Se&&r.method||n.method||"GET")).toUpperCase();this.params.method=i,this.body=n.body,this.txSize=ie(n.body)||0}function E(e,t){if(this.endTime=(0,o.t)(),this.params||(this.params={}),(0,Ee.iW)(this.params))return;let r;this.params.status=t?t.status:0,"string"==typeof this.rxSize&&this.rxSize.length>0&&(r=+this.rxSize);const n={txSize:this.txSize,rxSize:r,duration:(0,o.t)()-this.startTime},i=[this.params,n,this.startTime,this.endTime,"fetch"];this.targets.forEach(e=>A(i,this,e))}function T(e){const t=this.params,r=this.metrics;if(this.ended)return;this.ended=!0;for(let t=0;t<xe;t++)e.removeEventListener(Ae[t],this.listener,!1);if(t.aborted)return;if((0,Ee.iW)(t))return;r.duration=(0,o.t)()-this.startTime,this.loadCaptureCalled||4!==e.readyState?null==t.status&&(t.status=0):x(this,e),r.cbTime=this.cbTime;const n=[t,r,this.startTime,this.endTime,"xhr"];this.targets.forEach(e=>A(n,this,e))}function A(e,r,i){n("xhr",[...e,i],r,t.K7.ajax)}function x(e,n){e.params.status=n.status;var i=function(e,t){var r=e.responseType;return"json"===r&&null!==t?t:"arraybuffer"===r||"blob"===r||"json"===r?ie(e.response):"text"===r||""===r||void 0===r?ie(e.responseText):void 0}(n,e.lastSize);if(i&&(e.metrics.rxSize=i),e.sameOrigin&&n.getAllResponseHeaders().indexOf(Oe)>=0){var o=n.getResponseHeader(Oe);o&&((0,s.p)(O.rs,["Ajax/CrossApplicationTracing/Header/Seen"],void 0,t.K7.metrics,r),e.params.cat=o.split(", ").pop())}e.loadCaptureCalled=!0}r.on("new-xhr",a),r.on("open-xhr-start",c),r.on("open-xhr-end",d),r.on("send-xhr-start",u),r.on("xhr-cb-time",l),r.on("xhr-load-added",h),r.on("xhr-load-removed",p),r.on("xhr-resolved",g),r.on("addEventListener-end",m),r.on("removeEventListener-end",v),r.on("fn-end",b),r.on("fetch-before-start",w),r.on("fetch-start",R),r.on("fn-start",y),r.on("fetch-done",E)}(e,this.ee,this.handler,this.dt),this.importAggregator(e,()=>i.e(478).then(i.bind(i,3845)))}}function ke(e,t){var r=(0,be.D)(t),n=e.params||e;n.hostname=r.hostname,n.port=r.port,n.protocol=r.protocol,n.host=r.hostname+":"+r.port,n.pathname=r.pathname,e.parsedOrigin=r,e.sameOrigin=r.sameOrigin}const Ne={},je=["pushState","replaceState"];function De(e){const t=function(e){return(e||oe.ee).get("history")}(e);return!f.RI||Ne[t.debugId]++||(Ne[t.debugId]=1,(0,ae.YM)(t).inPlace(window.history,je,"-")),t}var Ce=i(3738);function Le(e){(0,c.Y)(a.BL,function(r=Date.now()){const n=r-f.WN;n<0&&(0,h.R)(62,r),(0,s.p)(O.XG,[a.BL,{time:n}],void 0,t.K7.metrics,e.ee),e.addToTrace({name:a.BL,start:r,origin:"nr"}),(0,s.p)(a.Pl+a.hG,[n,a.BL],void 0,t.K7.genericEvents,e.ee)},e)}const{He:Ie,bD:Me,d3:Be,Kp:He,TZ:Ke,Lc:We,uP:Fe,Rz:Ue}=Ce;class Ve extends y{static featureName=Ke;constructor(e){var r;super(e,Ke),r=e,(0,c.Y)(a.U2,function(e){if(!(e&&"object"==typeof e&&e.name&&e.start))return;const n={n:e.name,s:e.start-f.WN,e:(e.end||e.start)-f.WN,o:e.origin||"",t:"api"};n.s<0||n.e<0||n.e<n.s?(0,h.R)(61,{start:n.s,end:n.e}):(0,s.p)("bstApi",[n],void 0,t.K7.sessionTrace,r.ee)},r),Le(e);if(!(0,g.V)(e.init))return void this.deregisterDrain();const n=this.ee;let d;De(n),this.eventsEE=(0,se.u)(n),this.eventsEE.on(Fe,function(e,t){this.bstStart=(0,o.t)()}),this.eventsEE.on(We,function(e,r){(0,s.p)("bst",[e[0],r,this.bstStart,(0,o.t)()],void 0,t.K7.sessionTrace,n)}),n.on(Ue+Be,function(e){this.time=(0,o.t)(),this.startPath=location.pathname+location.hash}),n.on(Ue+He,function(e){(0,s.p)("bstHist",[location.pathname+location.hash,this.startPath,this.time],void 0,t.K7.sessionTrace,n)});try{d=new PerformanceObserver(e=>{const r=e.getEntries();(0,s.p)(Ie,[r],void 0,t.K7.sessionTrace,n)}),d.observe({type:Me,buffered:!0})}catch(e){}this.importAggregator(e,()=>i.e(478).then(i.bind(i,6974)),{resourceObserver:d})}}var ze=i(6344);class Ge extends y{static featureName=ze.TZ;#n;recorder;constructor(e){var r;let n;super(e,ze.TZ),r=e,(0,c.Y)(a.CH,function(){(0,s.p)(a.CH,[],void 0,t.K7.sessionReplay,r.ee)},r),function(e){(0,c.Y)(a.Tb,function(){(0,s.p)(a.Tb,[],void 0,t.K7.sessionReplay,e.ee)},e)}(e);try{n=JSON.parse(localStorage.getItem("".concat(w.H3,"_").concat(w.uh)))}catch(e){}(0,p.SR)(e.init)&&this.ee.on(a.CH,()=>this.#i()),this.#s(n)&&this.importRecorder().then(e=>{e.startRecording(ze.Qb.PRELOAD,n?.sessionReplayMode)}),this.importAggregator(this.agentRef,()=>i.e(478).then(i.bind(i,6167)),this),this.ee.on("err",e=>{this.blocked||this.agentRef.runtime.isRecording&&(this.errorNoticed=!0,(0,s.p)(ze.Vh,[e],void 0,this.featureName,this.ee))})}#s(e){return e&&(e.sessionReplayMode===w.g.FULL||e.sessionReplayMode===w.g.ERROR)||(0,p.Aw)(this.agentRef.init)}importRecorder(){return this.recorder?Promise.resolve(this.recorder):(this.#n??=Promise.all([i.e(478),i.e(249)]).then(i.bind(i,4866)).then(({Recorder:e})=>(this.recorder=new e(this),this.recorder)).catch(e=>{throw this.ee.emit("internal-error",[e]),this.blocked=!0,e}),this.#n)}#i(){this.blocked||(this.featAggregate?this.featAggregate.mode!==w.g.FULL&&this.featAggregate.initializeRecording(w.g.FULL,!0,ze.Qb.API):this.importRecorder().then(()=>{this.recorder.startRecording(ze.Qb.API,w.g.FULL)}))}}var Ye=i(3962);class Ze extends y{static featureName=Ye.TZ;constructor(e){if(super(e,Ye.TZ),function(e){const r=e.ee.get("tracer");function n(){}(0,c.Y)(a.dT,function(e){return(new n).get("object"==typeof e?e:{})},e);const i=n.prototype={createTracer:function(n,i){var a={},c=this,d="function"==typeof i;return(0,s.p)(O.xV,["API/createTracer/called"],void 0,t.K7.metrics,e.ee),function(){if(r.emit((d?"":"no-")+"fn-start",[(0,o.t)(),c,d],a),d)try{return i.apply(this,arguments)}catch(e){const t="string"==typeof e?new Error(e):e;throw r.emit("fn-err",[arguments,this,t],a),t}finally{r.emit("fn-end",[(0,o.t)()],a)}}}};["actionText","setName","setAttribute","save","ignore","onEnd","getContext","end","get"].forEach(r=>{c.Y.apply(this,[r,function(){return(0,s.p)(a.hw+r,[performance.now(),...arguments],this,t.K7.softNav,e.ee),this},e,i])}),(0,c.Y)(a.PA,function(){(0,s.p)(a.hw+"routeName",[performance.now(),...arguments],void 0,t.K7.softNav,e.ee)},e)}(e),!f.RI||!(0,T.dV)().o.MO)return;const r=De(this.ee);try{this.removeOnAbort=new AbortController}catch(e){}Ye.tC.forEach(e=>{(0,N.sp)(e,e=>{l(e)},!0,this.removeOnAbort?.signal)});const n=()=>(0,s.p)("newURL",[(0,o.t)(),""+window.location],void 0,this.featureName,this.ee);r.on("pushState-end",n),r.on("replaceState-end",n),(0,N.sp)(Ye.OV,e=>{l(e),(0,s.p)("newURL",[e.timeStamp,""+window.location],void 0,this.featureName,this.ee)},!0,this.removeOnAbort?.signal);let d=!1;const u=new((0,T.dV)().o.MO)((e,t)=>{d||(d=!0,requestAnimationFrame(()=>{(0,s.p)("newDom",[(0,o.t)()],void 0,this.featureName,this.ee),d=!1}))}),l=(0,m.s)(e=>{"loading"!==document.readyState&&((0,s.p)("newUIEvent",[e],void 0,this.featureName,this.ee),u.observe(document.body,{attributes:!0,childList:!0,subtree:!0,characterData:!0}))},100,{leading:!0});this.abortHandler=function(){this.removeOnAbort?.abort(),u.disconnect(),this.abortHandler=void 0},this.importAggregator(e,()=>i.e(478).then(i.bind(i,4393)),{domObserver:u})}}var qe=i(3333),Xe=i(9119);const $e={},Qe=new Set;function Je(e){return"string"==typeof e?{type:"string",size:(new TextEncoder).encode(e).length}:e instanceof ArrayBuffer?{type:"ArrayBuffer",size:e.byteLength}:e instanceof Blob?{type:"Blob",size:e.size}:e instanceof DataView?{type:"DataView",size:e.byteLength}:ArrayBuffer.isView(e)?{type:"TypedArray",size:e.byteLength}:{type:"unknown",size:0}}class et{constructor(e,t){this.timestamp=(0,o.t)(),this.currentUrl=(0,Xe.L)(window.location.href),this.socketId=(0,Y.LA)(8),this.requestedUrl=(0,Xe.L)(e),this.requestedProtocols=Array.isArray(t)?t.join(","):t||"",this.openedAt=void 0,this.protocol=void 0,this.extensions=void 0,this.binaryType=void 0,this.messageOrigin=void 0,this.messageCount=0,this.messageBytes=0,this.messageBytesMin=0,this.messageBytesMax=0,this.messageTypes=void 0,this.sendCount=0,this.sendBytes=0,this.sendBytesMin=0,this.sendBytesMax=0,this.sendTypes=void 0,this.closedAt=void 0,this.closeCode=void 0,this.closeReason="unknown",this.closeWasClean=void 0,this.connectedDuration=0,this.hasErrors=void 0}}class tt extends y{static featureName=qe.TZ;constructor(e){super(e,qe.TZ);const r=e.init.feature_flags.includes("websockets"),n=!e.init.feature_flags.includes("no_spv"),d=[e.init.page_action.enabled,e.init.performance.capture_marks,e.init.performance.capture_measures,e.init.performance.resources.enabled,e.init.user_actions.enabled,r,n];var u;let l;if(u=e,(0,c.Y)(a.hG,(e,t)=>U(e,t,u),u),function(e){(0,c.Y)(a.fF,(t,r)=>z(t,r,e),e)}(e),Le(e),Q(e),function(e){(0,c.Y)(a.V1,(t,r)=>V(t,r,e),e)}(e),this.removeOnAbort=new AbortController,this.abortHandler=()=>{this.removeOnAbort.abort(),this.abortHandler=void 0},r){const h=function(e){if(!(0,T.dV)().o.WS)return e;const t=e.get("websockets");if($e[t.debugId]++)return t;$e[t.debugId]=1,(0,x.G)(()=>{const e=(0,o.t)();Qe.forEach(r=>{r.nrData.closedAt=e,r.nrData.closeCode=1001,r.nrData.closeReason="Page navigating away",r.nrData.closeWasClean=!1,r.nrData.openedAt&&(r.nrData.connectedDuration=e-r.nrData.openedAt),t.emit("ws",[r.nrData],r)})});class r extends WebSocket{static name="WebSocket";static toString(){return"function WebSocket() { [native code] }"}toString(){return"[object WebSocket]"}get[Symbol.toStringTag](){return r.name}#o(e){(e.__newrelic??={}).socketId=this.nrData.socketId,this.nrData.hasErrors??=!0}constructor(...e){super(...e),this.nrData=new et(e[0],e[1]),this.addEventListener("open",()=>{this.nrData.openedAt=(0,o.t)(),["protocol","extensions","binaryType"].forEach(e=>{this.nrData[e]=this[e]}),Qe.add(this)}),this.addEventListener("message",e=>{const{type:t,size:r}=Je(e.data);this.nrData.messageOrigin??=(0,Xe.L)(e.origin),this.nrData.messageCount++,this.nrData.messageBytes+=r,this.nrData.messageBytesMin=Math.min(this.nrData.messageBytesMin||1/0,r),this.nrData.messageBytesMax=Math.max(this.nrData.messageBytesMax,r),(this.nrData.messageTypes??"").includes(t)||(this.nrData.messageTypes=this.nrData.messageTypes?"".concat(this.nrData.messageTypes,",").concat(t):t)}),this.addEventListener("close",e=>{this.nrData.closedAt=(0,o.t)(),this.nrData.closeCode=e.code,e.reason&&(this.nrData.closeReason=e.reason),this.nrData.closeWasClean=e.wasClean,this.nrData.connectedDuration=this.nrData.closedAt-this.nrData.openedAt,Qe.delete(this),t.emit("ws",[this.nrData],this)})}addEventListener(e,t,...r){const n=this,i="function"==typeof t?function(...e){try{return t.apply(this,e)}catch(e){throw n.#o(e),e}}:t?.handleEvent?{handleEvent:function(...e){try{return t.handleEvent.apply(t,e)}catch(e){throw n.#o(e),e}}}:t;return super.addEventListener(e,i,...r)}send(e){if(this.readyState===WebSocket.OPEN){const{type:t,size:r}=Je(e);this.nrData.sendCount++,this.nrData.sendBytes+=r,this.nrData.sendBytesMin=Math.min(this.nrData.sendBytesMin||1/0,r),this.nrData.sendBytesMax=Math.max(this.nrData.sendBytesMax,r),(this.nrData.sendTypes??"").includes(t)||(this.nrData.sendTypes=this.nrData.sendTypes?"".concat(this.nrData.sendTypes,",").concat(t):t)}try{return super.send(e)}catch(e){throw this.#o(e),e}}close(...e){try{super.close(...e)}catch(e){throw this.#o(e),e}}}return f.gm.WebSocket=r,t}(this.ee);h.on("ws",e=>{(0,s.p)("ws-complete",[e],void 0,this.featureName,this.ee)})}if(n&&f.gm.addEventListener("securitypolicyviolation",e=>{(0,s.p)("spv",[e],void 0,t.K7.genericEvents,this.ee)},(0,N.jT)(!1,this.removeOnAbort.signal)),f.RI){if(ye(this.ee,e),ue(this.ee,e),l=De(this.ee),e.init.user_actions.enabled){function p(t){const r=(0,be.D)(t);return e.beacons.includes(r.hostname+":"+r.port)}function g(){l.emit("navChange")}qe.Zp.forEach(e=>(0,N.sp)(e,e=>(0,s.p)("ua",[e],void 0,this.featureName,this.ee),!0)),qe.qN.forEach(e=>{const t=(0,m.s)(e=>{(0,s.p)("ua",[e],void 0,this.featureName,this.ee)},500,{leading:!0});(0,N.sp)(e,t)}),f.gm.addEventListener("error",()=>{(0,s.p)("uaErr",[],void 0,t.K7.genericEvents,this.ee)},(0,N.jT)(!1,this.removeOnAbort.signal)),this.ee.on("open-xhr-start",(e,r)=>{p(e[1])||r.addEventListener("readystatechange",()=>{2===r.readyState&&(0,s.p)("uaXhr",[],void 0,t.K7.genericEvents,this.ee)},(0,N.jT)(void 0,this.removeOnAbort.signal))}),this.ee.on("fetch-start",e=>{e.length>=1&&!p(Te(e[0]))&&(0,s.p)("uaXhr",[],void 0,t.K7.genericEvents,this.ee)}),l.on("pushState-end",g),l.on("replaceState-end",g),window.addEventListener("hashchange",g,(0,N.jT)(!0,this.removeOnAbort.signal)),window.addEventListener("popstate",g,(0,N.jT)(!0,this.removeOnAbort.signal))}if(e.init.performance.resources.enabled&&f.gm.PerformanceObserver?.supportedEntryTypes.includes("resource")){new PerformanceObserver(e=>{e.getEntries().forEach(e=>{(0,s.p)("browserPerformance.resource",[e],void 0,this.featureName,this.ee)})}).observe({type:"resource",buffered:!0})}}d.some(e=>e)?this.importAggregator(e,()=>i.e(478).then(i.bind(i,8019))):this.deregisterDrain()}}var rt=i(2646);const nt=new Map;function it(e,t,r,n,i=!0,s){if("object"!=typeof t||!t||"string"!=typeof r||!r||"function"!=typeof t[r])return(0,h.R)(29);const o=function(e){return(e||oe.ee).get("logger")}(e),a=(0,ae.YM)(o,void 0,s),c=new rt.y(oe.P);c.level=n.level,c.customAttributes=n.customAttributes,c.autoCaptured=i;const d=t[r]?.[ae.Jt]||t[r];return nt.set(d,c),a.inPlace(t,[r],"wrap-logger-",()=>nt.get(d),void 0,!0),o}var st=i(1910);class ot extends y{static featureName=K.TZ;constructor(e){var t;super(e,K.TZ),t=e,(0,c.Y)(a.$9,(e,r)=>F(e,r,t),t),function(e){(0,c.Y)(a.Wb,(t,r,{customAttributes:n={},level:i=K.p_.INFO}={})=>{it(e.ee,t,r,{customAttributes:n,level:i},!1,e)},e)}(e),Q(e);const r=this.ee;["log","error","warn","info","debug","trace"].forEach(t=>{(0,st.i)(f.gm.console[t]),it(r,f.gm.console,t,{level:"log"===t?"info":t},void 0,e)}),this.ee.on("wrap-logger-end",function([e],t,n,i=[]){const{level:s,customAttributes:o,autoCaptured:a}=this;i.forEach(t=>{(0,W.R)(r,e,o,s,a,t)})}),this.importAggregator(e,()=>i.e(478).then(i.bind(i,5288)))}}new A({features:[Pe,E,_,Ve,Ge,P,ee,tt,ot,Ze],loaderType:"spa"})})()})();</script>
<title>Georgia State University - Georgia State News Hub</title><meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="format-detection" content="telephone=no">
<!-- This site is optimized with the Yoast SEO Premium plugin v11.3 - https://yoast.com/wordpress/plugins/seo/ -->
<meta name="description" content="Featuring Georgia State&#039;s latest news, faculty and research experts and access to our story archive."/>
<link rel="canonical" href="https://news.gsu.edu/" />
<meta property="og:locale" content="en_US" />
<meta property="og:type" content="website" />
<meta property="og:title" content="Georgia State University - Georgia State News Hub" />
<meta property="og:description" content="Featuring Georgia State&#039;s latest news, faculty and research experts and access to our story archive." />
<meta property="og:url" content="https://news.gsu.edu/" />
<meta property="og:site_name" content="Georgia State News Hub" />
<script type='application/ld+json' class='yoast-schema-graph yoast-schema-graph--main'>{"@context":"https://schema.org","@graph":[{"@type":"Organization","@id":"https://news.gsu.edu/#organization","name":"Georgia State University","url":"https://news.gsu.edu/","sameAs":[]},{"@type":"WebSite","@id":"https://news.gsu.edu/#website","url":"https://news.gsu.edu/","name":"Georgia State News Hub","publisher":{"@id":"https://news.gsu.edu/#organization"},"potentialAction":{"@type":"SearchAction","target":"https://news.gsu.edu/?s={search_term_string}","query-input":"required name=search_term_string"}},{"@type":"WebPage","@id":"https://news.gsu.edu/#webpage","url":"https://news.gsu.edu/","inLanguage":"en-US","name":"Georgia State University - Georgia State News Hub","isPartOf":{"@id":"https://news.gsu.edu/#website"},"about":{"@id":"https://news.gsu.edu/#organization"},"datePublished":"2016-07-15T18:50:05+00:00","dateModified":"2026-05-01T16:09:56+00:00","description":"Featuring Georgia State's latest news, faculty and research experts and access to our story archive."}]}</script>
<!-- / Yoast SEO Premium plugin. -->

<link rel='dns-prefetch' href='//fonts.googleapis.com' />
<link rel='dns-prefetch' href='//s.w.org' />
<link rel="alternate" type="application/rss+xml" title="Georgia State News Hub &raquo; Feed" href="https://news.gsu.edu/feed/" />
		<script type="text/javascript">
			window._wpemojiSettings = {"baseUrl":"https:\/\/s.w.org\/images\/core\/emoji\/11\/72x72\/","ext":".png","svgUrl":"https:\/\/s.w.org\/images\/core\/emoji\/11\/svg\/","svgExt":".svg","source":{"concatemoji":"https:\/\/news.gsu.edu\/wp-includes\/js\/wp-emoji-release.min.js?ver=4.9.8"}};
			!function(a,b,c){function d(a,b){var c=String.fromCharCode;l.clearRect(0,0,k.width,k.height),l.fillText(c.apply(this,a),0,0);var d=k.toDataURL();l.clearRect(0,0,k.width,k.height),l.fillText(c.apply(this,b),0,0);var e=k.toDataURL();return d===e}function e(a){var b;if(!l||!l.fillText)return!1;switch(l.textBaseline="top",l.font="600 32px Arial",a){case"flag":return!(b=d([55356,56826,55356,56819],[55356,56826,8203,55356,56819]))&&(b=d([55356,57332,56128,56423,56128,56418,56128,56421,56128,56430,56128,56423,56128,56447],[55356,57332,8203,56128,56423,8203,56128,56418,8203,56128,56421,8203,56128,56430,8203,56128,56423,8203,56128,56447]),!b);case"emoji":return b=d([55358,56760,9792,65039],[55358,56760,8203,9792,65039]),!b}return!1}function f(a){var c=b.createElement("script");c.src=a,c.defer=c.type="text/javascript",b.getElementsByTagName("head")[0].appendChild(c)}var g,h,i,j,k=b.createElement("canvas"),l=k.getContext&&k.getContext("2d");for(j=Array("flag","emoji"),c.supports={everything:!0,everythingExceptFlag:!0},i=0;i<j.length;i++)c.supports[j[i]]=e(j[i]),c.supports.everything=c.supports.everything&&c.supports[j[i]],"flag"!==j[i]&&(c.supports.everythingExceptFlag=c.supports.everythingExceptFlag&&c.supports[j[i]]);c.supports.everythingExceptFlag=c.supports.everythingExceptFlag&&!c.supports.flag,c.DOMReady=!1,c.readyCallback=function(){c.DOMReady=!0},c.supports.everything||(h=function(){c.readyCallback()},b.addEventListener?(b.addEventListener("DOMContentLoaded",h,!1),a.addEventListener("load",h,!1)):(a.attachEvent("onload",h),b.attachEvent("onreadystatechange",function(){"complete"===b.readyState&&c.readyCallback()})),g=c.source||{},g.concatemoji?f(g.concatemoji):g.wpemoji&&g.twemoji&&(f(g.twemoji),f(g.wpemoji)))}(window,document,window._wpemojiSettings);
		</script>
		<style type="text/css">
img.wp-smiley,
img.emoji {
	display: inline !important;
	border: none !important;
	box-shadow: none !important;
	height: 1em !important;
	width: 1em !important;
	margin: 0 .07em !important;
	vertical-align: -0.1em !important;
	background: none !important;
	padding: 0 !important;
}
</style>
<style type="text/css">@font-face {font-family:Dosis;font-style:normal;font-weight:400;src:url(/cf-fonts/s/dosis/5.0.18/latin/400/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Dosis;font-style:normal;font-weight:400;src:url(/cf-fonts/s/dosis/5.0.18/vietnamese/400/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Dosis;font-style:normal;font-weight:400;src:url(/cf-fonts/s/dosis/5.0.18/latin-ext/400/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Dosis;font-style:normal;font-weight:500;src:url(/cf-fonts/s/dosis/5.0.18/latin-ext/500/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Dosis;font-style:normal;font-weight:500;src:url(/cf-fonts/s/dosis/5.0.18/vietnamese/500/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Dosis;font-style:normal;font-weight:500;src:url(/cf-fonts/s/dosis/5.0.18/latin/500/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Dosis;font-style:normal;font-weight:600;src:url(/cf-fonts/s/dosis/5.0.18/latin/600/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Dosis;font-style:normal;font-weight:600;src:url(/cf-fonts/s/dosis/5.0.18/latin-ext/600/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Dosis;font-style:normal;font-weight:600;src:url(/cf-fonts/s/dosis/5.0.18/vietnamese/600/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Dosis;font-style:normal;font-weight:700;src:url(/cf-fonts/s/dosis/5.0.18/latin/700/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Dosis;font-style:normal;font-weight:700;src:url(/cf-fonts/s/dosis/5.0.18/latin-ext/700/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Dosis;font-style:normal;font-weight:700;src:url(/cf-fonts/s/dosis/5.0.18/vietnamese/700/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:400;src:url(/cf-fonts/s/roboto/5.0.11/greek-ext/400/normal.woff2);unicode-range:U+1F00-1FFF;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:400;src:url(/cf-fonts/s/roboto/5.0.11/cyrillic/400/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:400;src:url(/cf-fonts/s/roboto/5.0.11/latin/400/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:400;src:url(/cf-fonts/s/roboto/5.0.11/vietnamese/400/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:400;src:url(/cf-fonts/s/roboto/5.0.11/latin-ext/400/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:400;src:url(/cf-fonts/s/roboto/5.0.11/greek/400/normal.woff2);unicode-range:U+0370-03FF;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:400;src:url(/cf-fonts/s/roboto/5.0.11/cyrillic-ext/400/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:700;src:url(/cf-fonts/s/roboto/5.0.11/latin/700/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:700;src:url(/cf-fonts/s/roboto/5.0.11/greek-ext/700/normal.woff2);unicode-range:U+1F00-1FFF;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:700;src:url(/cf-fonts/s/roboto/5.0.11/cyrillic-ext/700/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:700;src:url(/cf-fonts/s/roboto/5.0.11/cyrillic/700/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:700;src:url(/cf-fonts/s/roboto/5.0.11/latin-ext/700/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:700;src:url(/cf-fonts/s/roboto/5.0.11/vietnamese/700/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:700;src:url(/cf-fonts/s/roboto/5.0.11/greek/700/normal.woff2);unicode-range:U+0370-03FF;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:900;src:url(/cf-fonts/s/roboto/5.0.11/latin-ext/900/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:900;src:url(/cf-fonts/s/roboto/5.0.11/greek/900/normal.woff2);unicode-range:U+0370-03FF;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:900;src:url(/cf-fonts/s/roboto/5.0.11/greek-ext/900/normal.woff2);unicode-range:U+1F00-1FFF;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:900;src:url(/cf-fonts/s/roboto/5.0.11/cyrillic-ext/900/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:900;src:url(/cf-fonts/s/roboto/5.0.11/latin/900/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:900;src:url(/cf-fonts/s/roboto/5.0.11/vietnamese/900/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Roboto;font-style:normal;font-weight:900;src:url(/cf-fonts/s/roboto/5.0.11/cyrillic/900/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}</style>
<link rel='stylesheet' id='beeteam_front_fontawsome_css-css'  href='https://news.gsu.edu/wp-content/plugins/ultimate-layouts-vc/assets/front-end/fontawesome/css/font-awesome.min.css?ver=3.0.6' type='text/css' media='all' />
<link rel='stylesheet' id='beeteam_front_animate_css-css'  href='https://news.gsu.edu/wp-content/plugins/ultimate-layouts-vc/assets/front-end/animate.css?ver=3.0.6' type='text/css' media='all' />
<link rel='stylesheet' id='beeteam_front_slick_css-css'  href='https://news.gsu.edu/wp-content/plugins/ultimate-layouts-vc/assets/front-end/slick/slick.css?ver=3.0.6' type='text/css' media='all' />
<link rel='stylesheet' id='beeteam_front_loadawsome_css-css'  href='https://news.gsu.edu/wp-content/plugins/ultimate-layouts-vc/assets/front-end/loaders.css?ver=3.0.6' type='text/css' media='all' />
<link rel='stylesheet' id='beeteam_front_priority_css-css'  href='https://news.gsu.edu/wp-content/plugins/ultimate-layouts-vc/assets/front-end/priority-navigation/priority-nav-core.css?ver=3.0.6' type='text/css' media='all' />
<link rel='stylesheet' id='beeteam_front_hover_css-css'  href='https://news.gsu.edu/wp-content/plugins/ultimate-layouts-vc/assets/front-end/hover-css/hover.css?ver=3.0.6' type='text/css' media='all' />
<link rel='stylesheet' id='beeteam_front_pagination_css-css'  href='https://news.gsu.edu/wp-content/plugins/ultimate-layouts-vc/assets/front-end/pagination/pagination.css?ver=3.0.6' type='text/css' media='all' />
<link rel='stylesheet' id='beeteam_front_malihu_css-css'  href='https://news.gsu.edu/wp-content/plugins/ultimate-layouts-vc/assets/front-end/malihuscroll/jquery.mCustomScrollbar.min.css?ver=3.0.6' type='text/css' media='all' />
<link rel='stylesheet' id='formidable-css'  href='https://news.gsu.edu/wp-content/plugins/formidable/css/formidableforms668.css?ver=1271738' type='text/css' media='all' />
<link rel='stylesheet' id='gsu-flex-2-1-magazine-news-css'  href='https://news.gsu.edu/wp-content/themes/gsu-flex-2.1-magazine/style.css?ver=2.1.2' type='text/css' media='all' />
<link rel='stylesheet' id='la-icon-maneger-style-css'  href='https://news.gsu.edu/wp-content/uploads/la_icon_sets/style.min.css?ver=4.9.8' type='text/css' media='all' />
<link rel='stylesheet' id='mo_saml_admin_settings_style-css'  href='https://news.gsu.edu/wp-content/plugins/miniorange-saml-20-single-sign-on/includes/css/jquery.ui.css?ver=4.9.8' type='text/css' media='all' />
<link rel='stylesheet' id='rps-blog-info-styles-css'  href='https://news.gsu.edu/wp-content/plugins/rps-blog-info/rps-blog-info.css?ver=4.9.8' type='text/css' media='all' />
<link rel='stylesheet' id='essential-grid-plugin-settings-css'  href='https://news.gsu.edu/wp-content/plugins/essential-grid/public/assets/css/settings.css?ver=2.3.2' type='text/css' media='all' />
<style type="text/css">@font-face {font-family:Open Sans;font-style:normal;font-weight:300;src:url(/cf-fonts/s/open-sans/5.0.20/latin-ext/300/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:300;src:url(/cf-fonts/s/open-sans/5.0.20/hebrew/300/normal.woff2);unicode-range:U+0590-05FF,U+200C-2010,U+20AA,U+25CC,U+FB1D-FB4F;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:300;src:url(/cf-fonts/s/open-sans/5.0.20/cyrillic/300/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:300;src:url(/cf-fonts/s/open-sans/5.0.20/cyrillic-ext/300/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:300;src:url(/cf-fonts/s/open-sans/5.0.20/greek/300/normal.woff2);unicode-range:U+0370-03FF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:300;src:url(/cf-fonts/s/open-sans/5.0.20/greek-ext/300/normal.woff2);unicode-range:U+1F00-1FFF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:300;src:url(/cf-fonts/s/open-sans/5.0.20/vietnamese/300/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:300;src:url(/cf-fonts/s/open-sans/5.0.20/latin/300/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:400;src:url(/cf-fonts/s/open-sans/5.0.20/cyrillic/400/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:400;src:url(/cf-fonts/s/open-sans/5.0.20/greek/400/normal.woff2);unicode-range:U+0370-03FF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:400;src:url(/cf-fonts/s/open-sans/5.0.20/cyrillic-ext/400/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:400;src:url(/cf-fonts/s/open-sans/5.0.20/latin/400/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:400;src:url(/cf-fonts/s/open-sans/5.0.20/vietnamese/400/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:400;src:url(/cf-fonts/s/open-sans/5.0.20/latin-ext/400/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:400;src:url(/cf-fonts/s/open-sans/5.0.20/greek-ext/400/normal.woff2);unicode-range:U+1F00-1FFF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:400;src:url(/cf-fonts/s/open-sans/5.0.20/hebrew/400/normal.woff2);unicode-range:U+0590-05FF,U+200C-2010,U+20AA,U+25CC,U+FB1D-FB4F;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:600;src:url(/cf-fonts/s/open-sans/5.0.20/cyrillic/600/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:600;src:url(/cf-fonts/s/open-sans/5.0.20/latin-ext/600/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:600;src:url(/cf-fonts/s/open-sans/5.0.20/latin/600/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:600;src:url(/cf-fonts/s/open-sans/5.0.20/greek-ext/600/normal.woff2);unicode-range:U+1F00-1FFF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:600;src:url(/cf-fonts/s/open-sans/5.0.20/greek/600/normal.woff2);unicode-range:U+0370-03FF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:600;src:url(/cf-fonts/s/open-sans/5.0.20/vietnamese/600/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:600;src:url(/cf-fonts/s/open-sans/5.0.20/cyrillic-ext/600/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:600;src:url(/cf-fonts/s/open-sans/5.0.20/hebrew/600/normal.woff2);unicode-range:U+0590-05FF,U+200C-2010,U+20AA,U+25CC,U+FB1D-FB4F;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:700;src:url(/cf-fonts/s/open-sans/5.0.20/greek-ext/700/normal.woff2);unicode-range:U+1F00-1FFF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:700;src:url(/cf-fonts/s/open-sans/5.0.20/cyrillic-ext/700/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:700;src:url(/cf-fonts/s/open-sans/5.0.20/hebrew/700/normal.woff2);unicode-range:U+0590-05FF,U+200C-2010,U+20AA,U+25CC,U+FB1D-FB4F;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:700;src:url(/cf-fonts/s/open-sans/5.0.20/latin/700/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:700;src:url(/cf-fonts/s/open-sans/5.0.20/vietnamese/700/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:700;src:url(/cf-fonts/s/open-sans/5.0.20/cyrillic/700/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:700;src:url(/cf-fonts/s/open-sans/5.0.20/greek/700/normal.woff2);unicode-range:U+0370-03FF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:700;src:url(/cf-fonts/s/open-sans/5.0.20/latin-ext/700/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:800;src:url(/cf-fonts/s/open-sans/5.0.20/cyrillic-ext/800/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:800;src:url(/cf-fonts/s/open-sans/5.0.20/greek-ext/800/normal.woff2);unicode-range:U+1F00-1FFF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:800;src:url(/cf-fonts/s/open-sans/5.0.20/latin-ext/800/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:800;src:url(/cf-fonts/s/open-sans/5.0.20/cyrillic/800/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:800;src:url(/cf-fonts/s/open-sans/5.0.20/hebrew/800/normal.woff2);unicode-range:U+0590-05FF,U+200C-2010,U+20AA,U+25CC,U+FB1D-FB4F;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:800;src:url(/cf-fonts/s/open-sans/5.0.20/latin/800/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:800;src:url(/cf-fonts/s/open-sans/5.0.20/greek/800/normal.woff2);unicode-range:U+0370-03FF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:800;src:url(/cf-fonts/s/open-sans/5.0.20/vietnamese/800/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}</style>
<style type="text/css">@font-face {font-family:Raleway;font-style:normal;font-weight:100;src:url(/cf-fonts/s/raleway/5.0.16/vietnamese/100/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:100;src:url(/cf-fonts/s/raleway/5.0.16/latin/100/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:100;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic/100/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:100;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic-ext/100/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:100;src:url(/cf-fonts/s/raleway/5.0.16/latin-ext/100/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:200;src:url(/cf-fonts/s/raleway/5.0.16/vietnamese/200/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:200;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic-ext/200/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:200;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic/200/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:200;src:url(/cf-fonts/s/raleway/5.0.16/latin/200/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:200;src:url(/cf-fonts/s/raleway/5.0.16/latin-ext/200/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:300;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic/300/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:300;src:url(/cf-fonts/s/raleway/5.0.16/vietnamese/300/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:300;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic-ext/300/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:300;src:url(/cf-fonts/s/raleway/5.0.16/latin/300/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:300;src:url(/cf-fonts/s/raleway/5.0.16/latin-ext/300/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:400;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic-ext/400/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:400;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic/400/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:400;src:url(/cf-fonts/s/raleway/5.0.16/latin/400/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:400;src:url(/cf-fonts/s/raleway/5.0.16/latin-ext/400/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:400;src:url(/cf-fonts/s/raleway/5.0.16/vietnamese/400/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:500;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic-ext/500/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:500;src:url(/cf-fonts/s/raleway/5.0.16/vietnamese/500/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:500;src:url(/cf-fonts/s/raleway/5.0.16/latin/500/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:500;src:url(/cf-fonts/s/raleway/5.0.16/latin-ext/500/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:500;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic/500/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:600;src:url(/cf-fonts/s/raleway/5.0.16/latin-ext/600/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:600;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic/600/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:600;src:url(/cf-fonts/s/raleway/5.0.16/latin/600/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:600;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic-ext/600/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:600;src:url(/cf-fonts/s/raleway/5.0.16/vietnamese/600/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:700;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic-ext/700/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:700;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic/700/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:700;src:url(/cf-fonts/s/raleway/5.0.16/vietnamese/700/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:700;src:url(/cf-fonts/s/raleway/5.0.16/latin-ext/700/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:700;src:url(/cf-fonts/s/raleway/5.0.16/latin/700/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:800;src:url(/cf-fonts/s/raleway/5.0.16/vietnamese/800/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:800;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic/800/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:800;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic-ext/800/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:800;src:url(/cf-fonts/s/raleway/5.0.16/latin/800/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:800;src:url(/cf-fonts/s/raleway/5.0.16/latin-ext/800/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:900;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic/900/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:900;src:url(/cf-fonts/s/raleway/5.0.16/latin/900/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:900;src:url(/cf-fonts/s/raleway/5.0.16/cyrillic-ext/900/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:900;src:url(/cf-fonts/s/raleway/5.0.16/latin-ext/900/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Raleway;font-style:normal;font-weight:900;src:url(/cf-fonts/s/raleway/5.0.16/vietnamese/900/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}</style>
<link rel='stylesheet' id='tp-droid-serif-css'  href='https://fonts.googleapis.com/css?family=Droid+Serif%3A400%2C700&#038;ver=4.9.8' type='text/css' media='all' />
<link rel='stylesheet' id='tp-fontello-css'  href='https://news.gsu.edu/wp-content/plugins/essential-grid/public/assets/font/fontello/css/fontello.css?ver=2.3.2' type='text/css' media='all' />
<link rel='stylesheet' id='rs-plugin-settings-css'  href='https://news.gsu.edu/wp-content/plugins/revslider/public/assets/css/settings.css?ver=5.4.8.3' type='text/css' media='all' />
<style id='rs-plugin-settings-inline-css' type='text/css'>
#rs-demo-id {}
</style>
<link rel='stylesheet' id='netsposts_css-css'  href='https://news.gsu.edu/wp-content/plugins/network-posts-extended/css/net_posts_extended.css?ver=4.9.8' type='text/css' media='all' />
<link rel='stylesheet' id='netsposts_star_css-css'  href='https://news.gsu.edu/wp-content/plugins/network-posts-extended/css/fontawesome-stars.css?ver=4.9.8' type='text/css' media='all' />
<link rel='stylesheet' id='rs-typewriter-front-css'  href='https://news.gsu.edu/wp-content/plugins/revslider-typewriter-addon/public/assets/css/typewriter.css?ver=1.0.3' type='text/css' media='all' />
<style type="text/css">@font-face {font-family:Lato;font-style:normal;font-weight:300;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/300/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:300;src:url(/cf-fonts/s/lato/5.0.18/latin/300/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:400;src:url(/cf-fonts/s/lato/5.0.18/latin/400/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:400;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/400/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:700;src:url(/cf-fonts/s/lato/5.0.18/latin/700/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:700;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/700/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:italic;font-weight:300;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/300/italic.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:italic;font-weight:300;src:url(/cf-fonts/s/lato/5.0.18/latin/300/italic.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Lato;font-style:italic;font-weight:400;src:url(/cf-fonts/s/lato/5.0.18/latin/400/italic.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Lato;font-style:italic;font-weight:400;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/400/italic.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:italic;font-weight:700;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/700/italic.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:italic;font-weight:700;src:url(/cf-fonts/s/lato/5.0.18/latin/700/italic.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}</style>
<style type="text/css">@font-face {font-family:Oswald;font-style:normal;font-weight:400;src:url(/cf-fonts/s/oswald/5.0.18/cyrillic/400/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Oswald;font-style:normal;font-weight:400;src:url(/cf-fonts/s/oswald/5.0.18/cyrillic-ext/400/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Oswald;font-style:normal;font-weight:400;src:url(/cf-fonts/s/oswald/5.0.18/latin-ext/400/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Oswald;font-style:normal;font-weight:400;src:url(/cf-fonts/s/oswald/5.0.18/latin/400/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Oswald;font-style:normal;font-weight:400;src:url(/cf-fonts/s/oswald/5.0.18/vietnamese/400/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}</style>
<style type="text/css">@font-face {font-family:EB Garamond;font-style:normal;font-weight:400;src:url(/cf-fonts/s/eb-garamond/5.0.16/cyrillic-ext/400/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:EB Garamond;font-style:normal;font-weight:400;src:url(/cf-fonts/s/eb-garamond/5.0.16/greek-ext/400/normal.woff2);unicode-range:U+1F00-1FFF;font-display:swap;}@font-face {font-family:EB Garamond;font-style:normal;font-weight:400;src:url(/cf-fonts/s/eb-garamond/5.0.16/greek/400/normal.woff2);unicode-range:U+0370-03FF;font-display:swap;}@font-face {font-family:EB Garamond;font-style:normal;font-weight:400;src:url(/cf-fonts/s/eb-garamond/5.0.16/latin/400/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:EB Garamond;font-style:normal;font-weight:400;src:url(/cf-fonts/s/eb-garamond/5.0.16/cyrillic/400/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:EB Garamond;font-style:normal;font-weight:400;src:url(/cf-fonts/s/eb-garamond/5.0.16/vietnamese/400/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:EB Garamond;font-style:normal;font-weight:400;src:url(/cf-fonts/s/eb-garamond/5.0.16/latin-ext/400/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}</style>
<link rel='stylesheet' id='font-awesome-css'  href='https://news.gsu.edu/wp-content/plugins/types/vendor/toolset/toolset-common/res/lib/font-awesome/css/font-awesome.min.css?ver=4.7.0' type='text/css' media='screen' />
<link rel='stylesheet' id='gsu-responsive-style-css'  href='https://news.gsu.edu/wp-content/themes/gsu-flex-2.1-magazine/responsive.css?ver=2.1.2' type='text/css' media='all' />
<link rel='stylesheet' id='slick-css'  href='https://news.gsu.edu/wp-content/themes/gsu-flex-2.1-magazine/javascript/slick/slick.css?ver=2.1.2' type='text/css' media='all' />
<link rel='stylesheet' id='slick-theme-css'  href='https://news.gsu.edu/wp-content/themes/gsu-flex-2.1-magazine/javascript/slick/slick-theme.css?ver=2.1.2' type='text/css' media='all' />
<link rel='stylesheet' id='dashicons-css'  href='https://news.gsu.edu/wp-includes/css/dashicons.min.css?ver=4.9.8' type='text/css' media='all' />
<link rel='stylesheet' id='toolset_bootstrap_styles-css'  href='https://news.gsu.edu/wp-content/plugins/types/vendor/toolset/toolset-common/res/lib/bootstrap/css/bootstrap.css?ver=3.4.0.2' type='text/css' media='screen' />
<link rel='stylesheet' id='wpdreams-asp-basic-css'  href='https://news.gsu.edu/wp-content/plugins/ajax-search-pro/css/style.basic.css?ver=xaNzoc' type='text/css' media='all' />
<link rel='stylesheet' id='wpdreams-asp-chosen-css'  href='https://news.gsu.edu/wp-content/plugins/ajax-search-pro/css/chosen/chosen.css?ver=xaNzoc' type='text/css' media='all' />
<link rel='stylesheet' id='wpdreams-ajaxsearchpro-instances-css'  href='https://news.gsu.edu/files/asp_upload/style.instances.css?ver=xaNzoc' type='text/css' media='all' />
<link rel='stylesheet' id='easy-social-share-buttons-css'  href='https://news.gsu.edu/wp-content/plugins/easy-social-share-buttons3/assets/css/easy-social-share-buttons.min.css?ver=4.9.8' type='text/css' media='all' />
<link rel='stylesheet' id='easy-social-share-buttons-functional-pack-css'  href='https://news.gsu.edu/wp-content/plugins/essb-functional-buttons-pack/assets/essb-functional-pack.css?ver=4.9.8' type='text/css' media='all' />
<link rel='stylesheet' id='mediaelement-css'  href='https://news.gsu.edu/wp-includes/js/mediaelement/mediaelementplayer-legacy.min.css?ver=4.2.6-78496d1' type='text/css' media='all' />
<link rel='stylesheet' id='wp-mediaelement-css'  href='https://news.gsu.edu/wp-includes/js/mediaelement/wp-mediaelement.min.css?ver=4.9.8' type='text/css' media='all' />
<link rel='stylesheet' id='views-pagination-style-css'  href='https://news.gsu.edu/wp-content/plugins/wp-views/embedded/res/css/wpv-pagination.css?ver=2.8.2' type='text/css' media='all' />
<style id='views-pagination-style-inline-css' type='text/css'>
.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-default > span.wpv-sort-list,.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-default .wpv-sort-list-item {border-color: #cdcdcd;}.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-default .wpv-sort-list-item a {color: #444;background-color: #fff;}.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-default a:hover,.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-default a:focus {color: #000;background-color: #eee;}.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-default .wpv-sort-list-item.wpv-sort-list-current a {color: #000;background-color: #eee;}.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-grey > span.wpv-sort-list,.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-grey .wpv-sort-list-item {border-color: #cdcdcd;}.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-grey .wpv-sort-list-item a {color: #444;background-color: #eeeeee;}.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-grey a:hover,.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-grey a:focus {color: #000;background-color: #e5e5e5;}.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-grey .wpv-sort-list-item.wpv-sort-list-current a {color: #000;background-color: #e5e5e5;}.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-blue > span.wpv-sort-list,.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-blue .wpv-sort-list-item {border-color: #0099cc;}.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-blue .wpv-sort-list-item a {color: #444;background-color: #cbddeb;}.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-blue a:hover,.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-blue a:focus {color: #000;background-color: #95bedd;}.wpv-sort-list-dropdown.wpv-sort-list-dropdown-style-blue .wpv-sort-list-item.wpv-sort-list-current a {color: #000;background-color: #95bedd;}
</style>
<link rel='stylesheet' id='bwl-advanced-faq-theme-css'  href='https://news.gsu.edu/wp-content/plugins/bwl-advanced-faq-manager/css/faq-style.css?ver=1.7.6' type='text/css' media='all' />
<link rel='stylesheet' id='js_composer_front-css'  href='https://news.gsu.edu/wp-content/plugins/js_composer/assets/css/js_composer.min.css?ver=5.7' type='text/css' media='all' />
<link rel='stylesheet' id='bsf-Defaults-css'  href='https://news.gsu.edu/files/smile_fonts/Defaults/Defaults.css?ver=4.9.8' type='text/css' media='all' />
<link rel='stylesheet' id='bsf-Georgia-State-Icon-Pack-css'  href='https://news.gsu.edu/files/smile_fonts/Georgia-State-Icon-Pack/Georgia-State-Icon-Pack.css?ver=4.9.8' type='text/css' media='all' />
<link rel='stylesheet' id='ultimate-style-min-css'  href='https://news.gsu.edu/wp-content/plugins/Ultimate_VC_Addons/assets/min-css/ultimate.min.css?ver=3.18.0' type='text/css' media='all' />
<link rel='stylesheet' id='ult-icons-css'  href='https://news.gsu.edu/wp-content/plugins/Ultimate_VC_Addons/assets/css/icons.css?ver=3.18.0' type='text/css' media='all' />
<link rel='stylesheet' id='ultimate-vidcons-css'  href='https://news.gsu.edu/wp-content/plugins/Ultimate_VC_Addons/assets/fonts/vidcons.css?ver=4.9.8' type='text/css' media='all' />
<link rel='stylesheet' id='ultimate-selected-google-fonts-style-css'  href='https://fonts.googleapis.com/css?family=Lato:normal,100,100italic,300,300italic,italic,700,700italic,900italic|Oswald:normal,200,300,500,600,700|Cormorant+Garamond:normal,300,300italic,italic,500,500italic,600,600italic,700,700italic|Andada|Homemade+Apple|Abel|Advent+Pro:|Amiko:|Anaheim|Cambay:|Libre+Baskerville:normal,italic,700|Goblin+One|Open+Sans:|IM+Fell+DW+Pica:|IM+Fell+Double+Pica:|PT+Serif:|PT+Serif+Caption:|Poppins:normal,100,100italic,200,200italic,300,300italic,italic,500,500italic,600,600italic,700,700italic,800,800italic,900,900italic|Roboto+Slab:normal,100,300,700|Roboto:normal,300,300italic,italic,500,500italic,700,700italic,900,900italic|DM+Serif+Text:normal,italic|Satisfy|VT323|Racing+Sans+One|Amatic+SC:normal,700|Just+Another+Hand|Agbalumo&#038;subset=latin-ext,latin,cyrillic-ext,cyrillic,vietnamese,greek,devanagari,greek-ext,hebrew' type='text/css' media='all' />
<link rel='stylesheet' id='ul_bete_front_css-css'  href='https://news.gsu.edu/wp-content/plugins/ultimate-layouts-vc/assets/front-end/core.css?ver=3.0.6' type='text/css' media='all' />
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/ultimate-layouts-vc/assets/front-end/lazysizes.js?ver=3.0.6'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/superfly-menu/includes/vendor/looks_awesome/icon_manager/js/md5.js?ver=1.0,0'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/superfly-menu/includes/vendor/looks_awesome/icon_manager/js/util.js?ver=1.0,0'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/jquery/jquery.js?ver=1.12.4'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/jquery/jquery-migrate.min.js?ver=1.4.1'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/miniorange-saml-20-single-sign-on/includes/js/settings.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/essential-grid/public/assets/js/jquery.esgbox.min.js?ver=2.3.2'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/essential-grid/public/assets/js/jquery.themepunch.tools.min.js?ver=2.3.2'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/revslider/public/assets/js/jquery.themepunch.revolution.min.js?ver=5.4.8.3'></script>
<!--[if lt IE 9]>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/themes/genesis/lib/js/html5shiv.js?ver=3.7.3'></script>
<![endif]-->
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/jquery/ui/core.min.js?ver=1.11.4'></script>
<script type='text/javascript'>
var mejsL10n = {"language":"en","strings":{"mejs.install-flash":"You are using a browser that does not have Flash player enabled or installed. Please turn on your Flash player plugin or download the latest version from https:\/\/get.adobe.com\/flashplayer\/","mejs.fullscreen-off":"Turn off Fullscreen","mejs.fullscreen-on":"Go Fullscreen","mejs.download-video":"Download Video","mejs.fullscreen":"Fullscreen","mejs.time-jump-forward":["Jump forward 1 second","Jump forward %1 seconds"],"mejs.loop":"Toggle Loop","mejs.play":"Play","mejs.pause":"Pause","mejs.close":"Close","mejs.time-slider":"Time Slider","mejs.time-help-text":"Use Left\/Right Arrow keys to advance one second, Up\/Down arrows to advance ten seconds.","mejs.time-skip-back":["Skip back 1 second","Skip back %1 seconds"],"mejs.captions-subtitles":"Captions\/Subtitles","mejs.captions-chapters":"Chapters","mejs.none":"None","mejs.mute-toggle":"Mute Toggle","mejs.volume-help-text":"Use Up\/Down Arrow keys to increase or decrease volume.","mejs.unmute":"Unmute","mejs.mute":"Mute","mejs.volume-slider":"Volume Slider","mejs.video-player":"Video Player","mejs.audio-player":"Audio Player","mejs.ad-skip":"Skip ad","mejs.ad-skip-info":["Skip in 1 second","Skip in %1 seconds"],"mejs.source-chooser":"Source Chooser","mejs.stop":"Stop","mejs.speed-rate":"Speed Rate","mejs.live-broadcast":"Live Broadcast","mejs.afrikaans":"Afrikaans","mejs.albanian":"Albanian","mejs.arabic":"Arabic","mejs.belarusian":"Belarusian","mejs.bulgarian":"Bulgarian","mejs.catalan":"Catalan","mejs.chinese":"Chinese","mejs.chinese-simplified":"Chinese (Simplified)","mejs.chinese-traditional":"Chinese (Traditional)","mejs.croatian":"Croatian","mejs.czech":"Czech","mejs.danish":"Danish","mejs.dutch":"Dutch","mejs.english":"English","mejs.estonian":"Estonian","mejs.filipino":"Filipino","mejs.finnish":"Finnish","mejs.french":"French","mejs.galician":"Galician","mejs.german":"German","mejs.greek":"Greek","mejs.haitian-creole":"Haitian Creole","mejs.hebrew":"Hebrew","mejs.hindi":"Hindi","mejs.hungarian":"Hungarian","mejs.icelandic":"Icelandic","mejs.indonesian":"Indonesian","mejs.irish":"Irish","mejs.italian":"Italian","mejs.japanese":"Japanese","mejs.korean":"Korean","mejs.latvian":"Latvian","mejs.lithuanian":"Lithuanian","mejs.macedonian":"Macedonian","mejs.malay":"Malay","mejs.maltese":"Maltese","mejs.norwegian":"Norwegian","mejs.persian":"Persian","mejs.polish":"Polish","mejs.portuguese":"Portuguese","mejs.romanian":"Romanian","mejs.russian":"Russian","mejs.serbian":"Serbian","mejs.slovak":"Slovak","mejs.slovenian":"Slovenian","mejs.spanish":"Spanish","mejs.swahili":"Swahili","mejs.swedish":"Swedish","mejs.tagalog":"Tagalog","mejs.thai":"Thai","mejs.turkish":"Turkish","mejs.ukrainian":"Ukrainian","mejs.vietnamese":"Vietnamese","mejs.welsh":"Welsh","mejs.yiddish":"Yiddish"}};
</script>
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/mediaelement/mediaelement-and-player.min.js?ver=4.2.6-78496d1'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/mediaelement/mediaelement-migrate.min.js?ver=4.9.8'></script>
<script type='text/javascript'>
/* <![CDATA[ */
var _wpmejsSettings = {"pluginPath":"\/wp-includes\/js\/mediaelement\/","classPrefix":"mejs-","stretching":"responsive"};
/* ]]> */
</script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/Ultimate_VC_Addons/assets/min-js/modernizr-custom.min.js?ver=3.18.0'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/Ultimate_VC_Addons/assets/min-js/jquery-ui.min.js?ver=3.18.0'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/Ultimate_VC_Addons/assets/min-js/ultimate.min.js?ver=3.18.0'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/Ultimate_VC_Addons/assets/min-js/modal-all.min.js?ver=3.18.0'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/Ultimate_VC_Addons/assets/min-js/jparallax.min.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/Ultimate_VC_Addons/assets/min-js/vhparallax.min.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/Ultimate_VC_Addons/assets/min-js/ultimate_bg.min.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/Ultimate_VC_Addons/assets/min-js/mb-YTPlayer.min.js?ver=4.9.8'></script>
<link rel='https://api.w.org/' href='https://news.gsu.edu/wp-json/' />
<link rel="EditURI" type="application/rsd+xml" title="RSD" href="https://news.gsu.edu/xmlrpc.php?rsd" />
<link rel="wlwmanifest" type="application/wlwmanifest+xml" href="https://news.gsu.edu/wp-includes/wlwmanifest.xml" /> 
<meta name="generator" content="WordPress 4.9.8" />
<link rel="alternate" type="application/json+oembed" href="https://news.gsu.edu/wp-json/oembed/1.0/embed?url=https%3A%2F%2Fnews.gsu.edu%2F" />
<link rel="alternate" type="text/xml+oembed" href="https://news.gsu.edu/wp-json/oembed/1.0/embed?url=https%3A%2F%2Fnews.gsu.edu%2F&#038;format=xml" />
    <style type="text/css" media="screen">
      div.printfriendly a, div.printfriendly a:link, div.printfriendly a:hover, div.printfriendly a:visited, div.printfriendly a:focus {
        text-decoration: none;
        border: none;
        -webkit-box-shadow:none!important;
        box-shadow:none!important;
      }
    </style>
           <style type="text/css" media="screen">
          div.printfriendly {
            margin: 12px 12px 12px 12px;
            position: relative;
            z-index: 1000;
          }
          div.printfriendly a, div.printfriendly a:link, div.printfriendly a:visited {
            font-size: 14px;
            color: #3AAA11;
            vertical-align: bottom;
          }
          .printfriendly a {
            box-shadow:none;
          }
          .printfriendly a:hover {
            cursor: pointer;
          }
          .printfriendly a img  {
            border: none;
            padding:0;
            margin-right: 6px;
            box-shadow: none;
            -webkit-box-shadow: none;
            -moz-box-shadow: none;
          }
          .printfriendly a span{
            vertical-align: bottom;
          }
          .pf-alignleft {
            float: left;
          }
          .pf-alignright {
            float: right;
          }
          div.pf-aligncenter {
            display: block;
            margin-left: auto;
            margin-right: auto;
            text-align: center;
          }
        </style>
        <style type="text/css" media="print">
          .printfriendly {
            display: none;
          }
        </style>
		<script type="text/javascript">
			var ajaxRevslider;
			
			jQuery(document).ready(function() {
				// CUSTOM AJAX CONTENT LOADING FUNCTION
				ajaxRevslider = function(obj) {
				
					// obj.type : Post Type
					// obj.id : ID of Content to Load
					// obj.aspectratio : The Aspect Ratio of the Container / Media
					// obj.selector : The Container Selector where the Content of Ajax will be injected. It is done via the Essential Grid on Return of Content
					
					var content = "";

					data = {};
					
					data.action = 'revslider_ajax_call_front';
					data.client_action = 'get_slider_html';
					data.token = '5c7ec03173';
					data.type = obj.type;
					data.id = obj.id;
					data.aspectratio = obj.aspectratio;
					
					// SYNC AJAX REQUEST
					jQuery.ajax({
						type:"post",
						url:"https://news.gsu.edu/wp-admin/admin-ajax.php",
						dataType: 'json',
						data:data,
						async:false,
						success: function(ret, textStatus, XMLHttpRequest) {
							if(ret.success == true)
								content = ret.data;								
						},
						error: function(e) {
							console.log(e);
						}
					});
					
					 // FIRST RETURN THE CONTENT WHEN IT IS LOADED !!
					 return content;						 
				};
				
				// CUSTOM AJAX FUNCTION TO REMOVE THE SLIDER
				var ajaxRemoveRevslider = function(obj) {
					return jQuery(obj.selector+" .rev_slider").revkill();
				};

				// EXTEND THE AJAX CONTENT LOADING TYPES WITH TYPE AND FUNCTION
				var extendessential = setInterval(function() {
					if (jQuery.fn.tpessential != undefined) {
						clearInterval(extendessential);
						if(typeof(jQuery.fn.tpessential.defaults) !== 'undefined') {
							jQuery.fn.tpessential.defaults.ajaxTypes.push({type:"revslider",func:ajaxRevslider,killfunc:ajaxRemoveRevslider,openAnimationSpeed:0.3});   
							// type:  Name of the Post to load via Ajax into the Essential Grid Ajax Container
							// func: the Function Name which is Called once the Item with the Post Type has been clicked
							// killfunc: function to kill in case the Ajax Window going to be removed (before Remove function !
							// openAnimationSpeed: how quick the Ajax Content window should be animated (default is 0.3)
						}
					}
				},30);
			});
		</script>
		<script type="text/javascript">document.documentElement.className += " js";</script>
<link rel="icon" href="https://news.gsu.edu/wp-content/themes/gsu-flex-2.1-magazine/images/flame.png" />
<link rel="pingback" href="https://news.gsu.edu/xmlrpc.php" />
    <!-- Google Tag Manager -->
    <script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
    new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
    j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
    'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
    })(window,document,'script','dataLayer','GTM-W2ZN7TC');</script>
    <!-- End Google Tag Manager -->
                <style type="text/css">@font-face {font-family:Open Sans;font-style:normal;font-weight:300;src:url(/cf-fonts/s/open-sans/5.0.20/hebrew/300/normal.woff2);unicode-range:U+0590-05FF,U+200C-2010,U+20AA,U+25CC,U+FB1D-FB4F;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:300;src:url(/cf-fonts/s/open-sans/5.0.20/cyrillic/300/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:300;src:url(/cf-fonts/s/open-sans/5.0.20/cyrillic-ext/300/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:300;src:url(/cf-fonts/s/open-sans/5.0.20/greek/300/normal.woff2);unicode-range:U+0370-03FF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:300;src:url(/cf-fonts/s/open-sans/5.0.20/latin/300/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:300;src:url(/cf-fonts/s/open-sans/5.0.20/vietnamese/300/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:300;src:url(/cf-fonts/s/open-sans/5.0.20/greek-ext/300/normal.woff2);unicode-range:U+1F00-1FFF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:300;src:url(/cf-fonts/s/open-sans/5.0.20/latin-ext/300/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:400;src:url(/cf-fonts/s/open-sans/5.0.20/greek-ext/400/normal.woff2);unicode-range:U+1F00-1FFF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:400;src:url(/cf-fonts/s/open-sans/5.0.20/greek/400/normal.woff2);unicode-range:U+0370-03FF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:400;src:url(/cf-fonts/s/open-sans/5.0.20/cyrillic/400/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:400;src:url(/cf-fonts/s/open-sans/5.0.20/hebrew/400/normal.woff2);unicode-range:U+0590-05FF,U+200C-2010,U+20AA,U+25CC,U+FB1D-FB4F;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:400;src:url(/cf-fonts/s/open-sans/5.0.20/vietnamese/400/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:400;src:url(/cf-fonts/s/open-sans/5.0.20/latin/400/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:400;src:url(/cf-fonts/s/open-sans/5.0.20/latin-ext/400/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:400;src:url(/cf-fonts/s/open-sans/5.0.20/cyrillic-ext/400/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:700;src:url(/cf-fonts/s/open-sans/5.0.20/cyrillic/700/normal.woff2);unicode-range:U+0301,U+0400-045F,U+0490-0491,U+04B0-04B1,U+2116;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:700;src:url(/cf-fonts/s/open-sans/5.0.20/greek/700/normal.woff2);unicode-range:U+0370-03FF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:700;src:url(/cf-fonts/s/open-sans/5.0.20/vietnamese/700/normal.woff2);unicode-range:U+0102-0103,U+0110-0111,U+0128-0129,U+0168-0169,U+01A0-01A1,U+01AF-01B0,U+0300-0301,U+0303-0304,U+0308-0309,U+0323,U+0329,U+1EA0-1EF9,U+20AB;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:700;src:url(/cf-fonts/s/open-sans/5.0.20/greek-ext/700/normal.woff2);unicode-range:U+1F00-1FFF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:700;src:url(/cf-fonts/s/open-sans/5.0.20/latin/700/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:700;src:url(/cf-fonts/s/open-sans/5.0.20/hebrew/700/normal.woff2);unicode-range:U+0590-05FF,U+200C-2010,U+20AA,U+25CC,U+FB1D-FB4F;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:700;src:url(/cf-fonts/s/open-sans/5.0.20/latin-ext/700/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Open Sans;font-style:normal;font-weight:700;src:url(/cf-fonts/s/open-sans/5.0.20/cyrillic-ext/700/normal.woff2);unicode-range:U+0460-052F,U+1C80-1C88,U+20B4,U+2DE0-2DFF,U+A640-A69F,U+FE2E-FE2F;font-display:swap;}</style>
                                <style type="text/css">@font-face {font-family:Lato;font-style:normal;font-weight:300;src:url(/cf-fonts/s/lato/5.0.18/latin/300/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:300;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/300/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:400;src:url(/cf-fonts/s/lato/5.0.18/latin/400/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:400;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/400/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:700;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/700/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:700;src:url(/cf-fonts/s/lato/5.0.18/latin/700/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}</style>
                <meta name="generator" content="Powered by WPBakery Page Builder - drag and drop page builder for WordPress."/>
<!--[if lte IE 9]><link rel="stylesheet" type="text/css" href="https://news.gsu.edu/wp-content/plugins/js_composer/assets/css/vc_lte_ie9.min.css" media="screen"><![endif]--><style type="text/css">.essb_links a:hover, .essb_links a:focus { background: none !important; }.essb_links.essb_share .essb_link_facebook a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_facebook a:hover, .essb_links .essb_link_facebook a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_twitter a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_twitter a:hover, .essb_links .essb_link_twitter a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_google a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_google a:hover, .essb_links .essb_link_google a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_pinterest a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_pinterest a:hover, .essb_links .essb_link_pinterest a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_linkedin a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_linkedin a:hover, .essb_links .essb_link_linkedin a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_digg a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_digg a:hover, .essb_links .essb_link_digg a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_del a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_del a:hover, .essb_links .essb_link_del a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_stumbleupon a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_stumbleupon a:hover, .essb_links .essb_link_stumbleupon a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_tumblr a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_tumblr a:hover, .essb_links .essb_link_tumblr a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_vk a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_vk a:hover, .essb_links .essb_link_vk a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_print a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_print a:hover, .essb_links .essb_link_print a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_mail a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_mail a:hover, .essb_links .essb_link_mail a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_flattr a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_flattr a:hover, .essb_links .essb_link_flattr a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_reddit a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_reddit a:hover, .essb_links .essb_link_reddit a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_buffer a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_buffer a:hover, .essb_links .essb_link_buffer a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_love a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_love a:hover, .essb_links .essb_link_love a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_weibo a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_weibo a:hover, .essb_links .essb_link_weibo a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_pocket a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_pocket a:hover, .essb_links .essb_link_pocket a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_xing a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_xing a:hover, .essb_links .essb_link_xing a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_ok a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_ok a:hover, .essb_links .essb_link_ok a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_mwp a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_mwp a:hover, .essb_links .essb_link_mwp a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_more a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_more_dots a, .essb_links.essb_share .essb_link_less a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_more a:hover, .essb_links .essb_link_more a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_more_dots a:hover, .essb_links .essb_link_more_dots a:focus, .essb_links.essb_share .essb_link_less a:hover, .essb_links .essb_link_less a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_whatsapp a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_whatsapp a:hover, .essb_links .essb_link_whatsapp a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_meneame a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_meneame a:hover, .essb_links .essb_link_meneame a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_blogger a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_blogger a:hover, .essb_links .essb_link_blogger a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_amazon a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_amazon a:hover, .essb_links .essb_link_amazon a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_yahoomail a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_yahoomail a:hover, .essb_links .essb_link_yahoomail a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_gmail a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_gmail a:hover, .essb_links .essb_link_gmail a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_aol a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_aol a:hover, .essb_links .essb_link_aol a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_newsvine a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_newsvine a:hover, .essb_links .essb_link_newsvine a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_hackernews a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_hackernews a:hover, .essb_links .essb_link_hackernews a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_evernote a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_evernote a:hover, .essb_links .essb_link_evernote a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_myspace a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_myspace a:hover, .essb_links .essb_link_myspace a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_mailru a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_mailru a:hover, .essb_links .essb_link_mailru a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_viadeo a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_viadeo a:hover, .essb_links .essb_link_viadeo a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_line a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_line a:hover, .essb_links .essb_link_line a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_flipboard a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_flipboard a:hover, .essb_links .essb_link_flipboard a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_comments a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_comments a:hover, .essb_links .essb_link_comments a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_yummly a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_yummly a:hover, .essb_links .essb_link_yummly a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_sms a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_sms a:hover, .essb_links .essb_link_sms a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_viber a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_viber a:hover, .essb_links .essb_link_viber a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_telegram a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_telegram a:hover, .essb_links .essb_link_telegram a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_subscribe a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_subscribe a:hover, .essb_links .essb_link_subscribe a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_skype a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_skype a:hover, .essb_links .essb_link_skype a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_messenger a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_messenger a:hover, .essb_links .essb_link_messenger a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_kakaotalk a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_kakaotalk a:hover, .essb_links .essb_link_kakaotalk a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_share a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_share a:hover, .essb_links .essb_link_share a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_livejournal a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_livejournal a:hover, .essb_links .essb_link_livejournal a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_yammer a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_yammer a:hover, .essb_links .essb_link_yammer a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_meetedgar a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_meetedgar a:hover, .essb_links .essb_link_meetedgar a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_fintel a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_fintel a:hover, .essb_links .essb_link_fintel a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_mix a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_mix a:hover, .essb_links .essb_link_mix a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_instapaper a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_instapaper a:hover, .essb_links .essb_link_instapaper a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_prevpost a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_prevpost a:hover, .essb_links .essb_link_prevpost a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_nextpost a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_nextpost a:hover, .essb_links .essb_link_nextpost a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_copylink a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_copylink a:hover, .essb_links .essb_link_copylink a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_bookmark a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_bookmark a:hover, .essb_links .essb_link_bookmark a:focus { background-color:#0071ce!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_qrcode a { background-color:#0034a9!important;color:#ffffff!important;}.essb_links.essb_share .essb_link_qrcode a:hover, .essb_links .essb_link_qrcode a:focus { background-color:#0071ce!important;color:#ffffff!important;}/*.single-coronavirus-expert .essb_links { top: 270px; position: absolute; right: 325px; } .single-coronavirus-expert .essb_links.essb_template_grey-round-retina li a, .essb_links.essb_template_grey-round-retina li a:focus, .single-coronavirus-expert .essb_links.essb_template_grey-round-retina li a:focus { background-color: #0034a9 !important; } .single-coronavirus-expert .essb_links.essb_template_grey-round-retina li a:hover { background-color: #00aeef !important; } .single-coronavirus-expert .phone { color:#0034a9; }*/ .essb_icon.essb_icon_twitter:before { content: url('data:image/svg+xml,<svg width="16" height="16" viewBox="0 0 1200 1227" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M714.163 519.284L1160.89 0H1055.03L667.137 450.887L357.328 0H0L468.492 681.821L0 1226.37H105.866L515.491 750.218L842.672 1226.37H1200L714.137 519.284H714.163ZM569.165 687.828L521.697 619.934L144.011 79.6944H306.615L611.412 515.685L658.88 583.579L1055.08 1150.3H892.476L569.165 687.854V687.828Z" fill="white"/></svg>'); }</style>
<script type="text/javascript">var essb_settings = {"ajax_url":"https:\/\/news.gsu.edu\/wp-admin\/admin-ajax.php","essb3_nonce":"b75e947167","essb3_plugin_url":"https:\/\/news.gsu.edu\/wp-content\/plugins\/easy-social-share-buttons3","essb3_facebook_total":true,"essb3_admin_ajax":false,"essb3_internal_counter":false,"essb3_counter_button_min":0,"essb3_counter_total_min":0,"essb3_no_counter_mailprint":false,"essb3_single_ajax":false,"ajax_type":"wp","twitter_counter":"self","google_counter":"","essb3_stats":false,"essb3_ga":false,"essb3_ga_mode":"simple","blog_url":"https:\/\/news.gsu.edu\/","essb3_postfloat_stay":false,"post_id":682};</script><style type='text/css'>.ac-container label{ color: #000000;
                       font-size: 18px;
                       background: #FFFFFF;
                       background: linear-gradient(#FFFFFF ,#EAEAEA);}.ac-container label:hover{
                            background: #FFFFFF;
                            color: #777777;
                    }.ac-container input:checked + label,
                   .ac-container input:checked + label:hover{
                            background: #FFFFFF;
                            color: #000000;
                      }.ac-container input:checked + label{
                      }.ac-container label:before, .ac-container label:after{
                        color: #000000;
                      }#baf_page_navigation .active_page{
                            background: #FFFFFF;
                            color: #000000 !important;
                    }div.baf-ctrl-btn span.baf-expand-all, div.baf-ctrl-btn span.baf-collapsible-all{
                            background: #FFFFFF;
                            color: #000000;
                    }div.baf-ctrl-btn span.baf-expand-all:hover, div.baf-ctrl-btn span.baf-collapsible-all:hover{
                            background: #EAEAEA;
                            color: #000000;
                    }.ac-container .bwl-faq-container article div,
                    .ac-container .bwl-faq-container article p {
                            font-size: 18px;
                   }.bwl-faq-wrapper ul.bwl-faq-tabs li.active{                            
                            border-color: #2C2C2C;
                   }.baf_taxonomy_info_container h2 {
    font-weight: 400;
    margin: 40px 10px 0px 10px;
    font-size: 3.6rem;
    line-height: 4.2rem;
    font-family: Lato;
    border-bottom: 2px dotted #0034a6;
    padding-bottom: 15px;
}
.ac-container label {  
	  font-weight: 400;
	  padding: 7px 35px 7px 20px;
	  line-height:1.5em;
}.ac-container .bwl-faq-search-panel span.baf-btn-clear{
                    right: 3px;
              }</style><script type="text/javascript">var baf_rtl_status = 0,
                                               first_color = '#FFFFFF',   
                                               checked_background = '#FF7B0D',
                                               hover_background = '#FFFFFF',
                                               bwl_advanced_faq_collapsible_accordion_status = '1',
                                               text_nothing_found = 'Nothing Found !',
                                               text_faqs = 'FAQs',
                                               text_faq = 'FAQ',                                               
                                               second_color = '#EAEAEA'</script>    <script type="text/javascript">
        
         var ajaxurl = "https://news.gsu.edu/wp-admin/admin-ajax.php",
                   err_faq_category = "Select FAQ Category!",
                   err_faq_captcha = " Incorrect Captcha Value!",
                   string_total = "Total",
//                   string_singular_page = "Page !",
                   string_singular_page = "Page !",
                   string_plural_page = "Pages !",
                   string_please_wait = "Please Wait .....",
                   string_ques_added = "Question successfully added for review!",
                   string_ques_unable_add = "Unable to add faq. Please try again!";
           
           var $noting_found_text = "Nothing Found!",
                $found_text = "Found",
                $singular_faq = "FAQ !",
                $plural_faq = "FAQs !";
       
    </script>

<meta name="generator" content="Powered by Slider Revolution 5.4.8.3 - responsive, Mobile-Friendly Slider Plugin for WordPress with comfortable drag and drop interface." />
            <style type="text/css">
                <!--
                @font-face {
                    font-family: 'asppsicons2';
                    src: url('https://news.gsu.edu/wp-content/plugins/ajax-search-pro/css/fonts/icons/icons2.eot');
                    src: url('https://news.gsu.edu/wp-content/plugins/ajax-search-pro/css/fonts/icons/icons2.eot?#iefix') format('embedded-opentype'),
                    url('https://news.gsu.edu/wp-content/plugins/ajax-search-pro/css/fonts/icons/icons2.woff2') format('woff2'),
                    url('https://news.gsu.edu/wp-content/plugins/ajax-search-pro/css/fonts/icons/icons2.woff') format('woff'),
                    url('https://news.gsu.edu/wp-content/plugins/ajax-search-pro/css/fonts/icons/icons2.ttf') format('truetype'),
                    url('https://news.gsu.edu/wp-content/plugins/ajax-search-pro/css/fonts/icons/icons2.svg#icons') format('svg');
                    font-weight: normal;
                    font-style: normal;
                }
                .asp_m{height: 0;}                -->
            </style>
                        <script type="text/javascript">
                if ( typeof _ASP !== "undefined" && _ASP !== null && typeof _ASP.initialize !== "undefined" )
                    _ASP.initialize();
            </script>
            

<!-- CJT Global Block (181) - Podcast Players - START -->
<style>
.ssp-wave-form {
    margin-left:20px!important;
}
</style>
<!-- CJT Global Block (181) - Podcast Players - END -->



<!-- CJT Global Block (130) - Sitewide CSS - START -->
<style>
.vc_grid p {
  line-height: 1.5;
}
p {
    color: #000;
}
li.widget.posts-in-sidebar {
  border-bottom: 0;
}
.term-rcb-hospitality-ms .breadcrumb, .tag-campus-news-faculty-staff .breadcrumb {
  display: none;
}
[id] {
  scroll-margin-top: 160px;
}
/* Uppercase Oswald */
h1[style*="Oswald"],h2[style*="Oswald"],h3[style*="Oswald"],h4[style*="Oswald"],p[style*="Oswald"],span[style*="Oswald"],div[style*="Oswald"] {
  text-transform: uppercase;
}
@supports (-webkit-touch-callout: inherit) {
   .upb_row_bg.vcpb-vz-jquery, .upb_row_bg.vcpb-default {
      background-attachment: scroll !important;
      background-size: auto 100vh !important;
   }
}
.media-contact p.media-contact-info {
    line-height:1em;
}
.vc_basic_grid.news-feature .vc_gitem-post-data-source-post_categories, 
.vc_basic_grid.news-latest .vc_gitem-post-data-source-post_categories {
    display:inline-block;
    line-height:1.2;
}
/* Option Weight */
option {
    font-weight:400;
}
/* Class Notes Expport View */
#classnote p {
    font-size:14px;
    line-height:1;
}
ul#lwe li {
    list-style-type: none;
}
/* Style LOAD MORE */
a.vc_general.vc_btn3.vc_btn3-size-md.vc_btn3-shape-square.vc_btn3-style-flat.vc_btn3-color-blue {
    background: #0034a9;
    font-size: 16px !important;
    font-weight: 400 !important;
    border: 1px solid #0034a9 !important;
}
/*  Style Carousel Arrows*/
.vc_grid.vc_grid-owl-theme .vc_grid-owl-nav.vc_grid-owl-nav-color-blue {
    color: #0034a9!important;
    position: relative !important;
}
.vc_grid.vc_grid-owl-theme .vc_grid-owl-nav {
    opacity: 1;
}
.vc_grid.vc_grid-owl-theme .vc_grid-owl-nav .vc_grid-owl-next, .vc_grid.vc_grid-owl-theme .vc_grid-owl-nav .vc_grid-owl-prev {
    opacity: 1 !important;
    font-size: 40px!important;
    line-height: 40px!important;
    height: 40px!important;
}
.vc_grid.vc_grid-owl-theme .vc_grid-owl-nav .vc_grid-owl-next:hover, .vc_grid.vc_grid-owl-theme .vc_grid-owl-nav .vc_grid-owl-prev:hover {
    opacity: .75!important;
    color: #0071EE;
}
#hgrid .vc_grid.vc_grid-owl-theme .vc_grid-owl-nav.vc_grid-owl-nav-color-blue {
    bottom: 10px !important;
}
#vgrid .vc_grid.vc_grid-owl-theme .vc_grid-owl-nav.vc_grid-owl-nav-color-blue {
    color: #0034a9 !important;
    bottom: -10px !important;
    margin-bottom:40px !important;
}
#vgrid .vc_grid.vc_grid-owl-theme .vc_grid-owl-nav .vc_grid-owl-next {
    left: 50% !important;
}
#vgrid .vc_grid.vc_grid-owl-theme .vc_grid-owl-nav .vc_grid-owl-prev {
    right: 50% !important;
}
#vgrid .vc_grid.vc_grid-owl-theme .vc_grid-owl-nav {
    text-align:center;
}
/* Fix line height in ordered and unordered lists. */
.site-inner li {
    line-height: 23px;
}
/* Adjust line heigth of menu items that wrap */
.nav-primary .genesis-nav-menu>li, .nav-primary .genesis-nav-menu {
    min-height: 40px;
    
}
/* Remove vertical lines on mobile Requires styles to be added to WPBakery Elments accoridngly */
@media (max-width: 768px) {
    #action-row .action-col .vc_column-inner,
    .action-row .action-col .vc_column-inner {
        border-left: none !important;border-top: 2px dotted #0034a9;padding-left: 15px !important;
    `   padding-top: 20px !important;
    }
    #action-row .inside-action-col .vc_column-inner,
    .action-row .inside-action-col .vc_column-inner {
        border-top: none !important;
    }
    #dotted-col-divider .right-col-contact .vc_column-inner,
    .dotted-col-divider .right-col-contact .vc_column-inner{
        border-top: none !important;    border-left: none !important;    padding-left: 15px !important;
    }
}
/* Reorder columns on mobile */
@media all and (max-width: 980px) {
    /*** wrap row in a flex box ***/
    .reverse_row {
        display: -webkit-box;
        display: -moz-box;
        display: -ms-flexbox;
        display: -webkit-flex;
        display: flex;
        -webkit-flex-wrap: wrap;
        /* Safari 6.1+ */
        flex-wrap: wrap;
    }
    /*** custom classes that will designate the order of columns in the flex box row ***/
    .first-on-mobile {
        -webkit-order: 1;
        order: 1;
    }
    .second-on-mobile {
        -webkit-order: 2;
        order: 2;
    }
    .third-on-mobile {
        -webkit-order: 3;
        order: 3;
    }
    .fourth-on-mobile {
        -webkit-order: 4;
        order: 4;
    }
    .fifth-on-mobile {
        -webkit-order: 5;
        order: 5;
    }
    .sixth-on-mobile {
        -webkit-order: 6;
        order: 6;
    }
}
/*News Site Specific styles */
/* experts */
body.single-expert .expert-col.ex-one {
    width: 300px;
}
body.single-expert .expert-col.ex-two {
    width: 615px;
}
/* Other */
.profile-excerpt p {
    font-size: 1.6rem;
    line-height: 2.0rem;
}
.lwe .lwd {
    font-size: 15px !important;
    line-height: 20px !important;
    font-weight: 400;
}
hr {
  border: 0;
  border-collapse: collapse;
  border-top: 1px solid #ddd;
  clear: none;
  margin: 1em 0;
}
blockquote {
    margin: 40px;
    border-left: 4px solid #eee;
    padding-left: 15px;
    margin-left: 0px;
}
blockquote, cite, i {
    font-style: italic;
}
blockquote::before {
    display:none;
}
.posts-in-sidebar h3.widgettitle.widget-title {
    font-size: 3.6rem;
    line-height: 39px;
    font-weight: 400;
}

.btn-info {
  color: #fff;
  text-shadow: 0 0px 0 rgba(255, 255, 255, 0.5);
  background-color: #0034a6;
  background-image: none;
  border-color: #fff;
  border-radius: 0;
  padding-top:18px;
  padding-bottom:18px;
  text-transform: uppercase;
  font-weight: 400;
}
.textwidget .btn-info {
    width: 100%;
}
.btn-info:hover,
.btn-info:focus,
.btn-info:active,
.btn-info.active,
.btn-info.disabled,
.btn-info[disabled] {
  color: #0034a6;
  background-color: #fff;
 border-color: #0034a6;
}
.btn-info:active,
.btn-info.active {
  background-color: #0034a6 \9;
}
.btn-info [class^="icon-"],
.btn-info [class*=" icon-"] {
  display: none;
}
.unit-newspage-link
 a {
     text-transform: uppercase;
     font-weight: 600 !important;
     font-family: Oswald !important;
     font-size: 16px !important;
     line-height: 21px !important;
     color: #cc0000;
 }
.unit-newspage-link {
    margin-bottom: 15px;
}
@media print {
    a[href]:after {
        content: none !important;
    }
    .title-area, #footer {
        display: none !important;
    }
    h2.entry-title:before {
    content: 'Georgia State University News Hub \A ';
    display:block;
    font-size: 24px;
    font-weight: normal;
    margin-bottom:30px;
    margin-top:-40px;
    }
    .essb_bottombar {
        display:none;
    }
    a { text-decoration: none; }
}
</style>
<!-- CJT Global Block (130) - Sitewide CSS - END -->



<!-- CJT Global Block (1) - GTM Body - START -->
<!-- Google Tag Manager (noscript) -->
<noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-PVD4M8V"
height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
<!-- End Google Tag Manager (noscript) -->
<!-- CJT Global Block (1) - GTM Body - END -->



<!-- CJT Global Block (3) - WordPress Captions - START -->
<style>
.wp-caption-text {
    color: #383838;
    line-height: 17px;
    font-weight:400;
    text-align:left;
}
.gallery-caption {
    margin-left: 5px !important;
}
.gallery img {
    border: 0px solid #ddd; 
}
.wp-caption.alignright {
    padding-left: 10px;
}
@media (max-width:800px) {
    aside.pull-quote {
    width: 100% !important;
    margin-bottom: 15px;
   }
}
</style>
<!-- CJT Global Block (3) - WordPress Captions - END -->



<!-- CJT Global Block (32) - GTM - News - GTM-PVD4M8V - START -->
<!-- Google Tag Manager -->
<script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','GTM-PVD4M8V');</script>
<!-- End Google Tag Manager -->
<!-- CJT Global Block (32) - GTM - News - GTM-PVD4M8V - END -->



<!-- CJT Global Block (39) - VC Post Grid Styles - START -->
<style>
ul.vc_grid-filter {
    margin-left: 0;
    margin-bottom: 30px !important;
}
.vc_grid-pagination .vc_grid-pagination-list > li.vc_grid-page > a,
.vc_grid-pagination .vc_grid-pagination-list > li.vc_grid-next > a, 
.vc_grid-pagination .vc_grid-pagination-list > li.vc_grid-prev > a, 
.vc_grid-pagination .vc_grid-pagination-list > li.vc_grid-first > a, 
.vc_grid-pagination .vc_grid-pagination-list > li.vc_grid-last > a {
    padding: 3px 10px !important;
    font-size: 16px !important;
    background-color: #0034a9 !important;
    border-color: #0034a9 !important;
    color: #fff !important;
    font-weight: 400;
    opacity: 1;
}
.vc_grid-pagination .vc_grid-pagination-list > li.vc_grid-page.vc_grid-active  > a,
.vc_grid-pagination .vc_grid-pagination-list > li.vc_grid-page  > a:hover {
    padding: 3px 10px !important;
    font-size: 16px !important;
    background-color: #fff !important;
    border-color: #0034a9 !important;
    color: #0034a9 !important;
    font-weight: 400;
    opacity: 1;
}
.vc_grid-pagination .vc_grid-pagination-list > li.vc_grid-next > a, 
.vc_grid-pagination .vc_grid-pagination-list > li.vc_grid-prev > a,
.vc_grid-pagination .vc_grid-pagination-list > li.vc_grid-first > a,
.vc_grid-pagination .vc_grid-pagination-list > li.vc_grid-last > a {
    font-size: 15px !important;
}
.vc_grid-filter.vc_grid-filter-filled-rounded>.vc_grid-filter-item:first-child, 
.vc_grid-filter.vc_grid-filter-filled>.vc_grid-filter-item:first-child {
    border-top-left-radius: 0px !important;
    border-bottom-left-radius: 0 !important;
}
.vc_grid-filter.vc_grid-filter-filled-rounded>.vc_grid-filter-item:last-child, 
.vc_grid-filter.vc_grid-filter-filled>.vc_grid-filter-item:last-child {
    border-top-right-radius: 0px !important;
    border-bottom-right-radius: 0 !important;
}
</style>
<!-- CJT Global Block (39) - VC Post Grid Styles - END -->



<!-- CJT Global Block (45) - Research Magazine Notes Grid - START -->
<style>
.eg-faculty-noteworthy-wrapper .esg-entry-media-wrapper {
    display: none !important;
}
a.eg-faculty-noteworthy-element-6 {
    display: inline;
    padding: 0;
    margin: 0;
}
.eg-faculty-noteworthy-element-6-a p {
    font-size: 16px;
    line-height: 21px;
    padding: 0 10px;
}
.eg-faculty-noteworthy-element-27 {
    padding-left: 10px !important;
    max-width: 240px;
}
.ubtn-ctn-left.noteworthy-modal {
    margin-bottom: 10px;
}
</style> 
<!-- CJT Global Block (45) - Research Magazine Notes Grid - END -->



<!-- CJT Global Block (53) - SiteImprove - START -->
<script type="text/javascript">
/*<![CDATA[*/
(function() {
    var sz = document.createElement('script'); sz.type = 'text/javascript'; sz.async = true;
    sz.src = '//siteimproveanalytics.com/js/siteanalyze_66357174.js';
    var s = document.getElementsByTagName('script')[0]; s.parentNode.insertBefore(sz, s);
})();
/*]]>*/
</script>
<!-- CJT Global Block (53) - SiteImprove - END -->

<script type="text/javascript">function setREVStartSize(e){									
						try{ e.c=jQuery(e.c);var i=jQuery(window).width(),t=9999,r=0,n=0,l=0,f=0,s=0,h=0;
							if(e.responsiveLevels&&(jQuery.each(e.responsiveLevels,function(e,f){f>i&&(t=r=f,l=e),i>f&&f>r&&(r=f,n=e)}),t>r&&(l=n)),f=e.gridheight[l]||e.gridheight[0]||e.gridheight,s=e.gridwidth[l]||e.gridwidth[0]||e.gridwidth,h=i/s,h=h>1?1:h,f=Math.round(h*f),"fullscreen"==e.sliderLayout){var u=(e.c.width(),jQuery(window).height());if(void 0!=e.fullScreenOffsetContainer){var c=e.fullScreenOffsetContainer.split(",");if (c) jQuery.each(c,function(e,i){u=jQuery(i).length>0?u-jQuery(i).outerHeight(!0):u}),e.fullScreenOffset.split("%").length>1&&void 0!=e.fullScreenOffset&&e.fullScreenOffset.length>0?u-=jQuery(window).height()*parseInt(e.fullScreenOffset,0)/100:void 0!=e.fullScreenOffset&&e.fullScreenOffset.length>0&&(u-=parseInt(e.fullScreenOffset,0))}f=u}else void 0!=e.minHeight&&f<e.minHeight&&(f=e.minHeight);e.c.closest(".rev_slider_wrapper").css({height:f})					
						}catch(d){console.log("Failure at Presize of Slider:"+d)}						
					};</script>
<style type="text/css" data-type="vc_custom-css">.experts-home .vc_custom_heading.vc_gitem-post-data.vc_gitem-post-data-source-post_title {
    margin-bottom: -5px;
}
.vc_basic_grid.news-latest, .vc_basic_grid.mobile-news-latest {
    padding:0 !important;
}
.vc_grid-filter.vc_grid-filter-size-xs .vc_grid-filter-item {
    font-size: 18px !important;
}
.vc_grid-filter.vc_grid-filter-filled > .vc_grid-filter-item:nth-child(n+1) {
    border-left: none;
    margin-left: 5px;
    border-top: none;
    margin-top: 5px;
}</style><style type="text/css" data-type="vc_shortcodes-custom-css">.vc_custom_1647455230012{margin-bottom: 0px !important;}.vc_custom_1468846886472{margin-bottom: 0px !important;background-color: #ffffff !important;}.vc_custom_1587477528698{margin-bottom: 15px !important;}.vc_custom_1619030632556{margin-bottom: 0px !important;}.vc_custom_1777648655677{border-top-width: 5px !important;}.vc_custom_1777651768608{margin-top: 30px !important;}.vc_custom_1591979182225{border-top-width: 5px !important;}.vc_custom_1652211028947{margin-bottom: 15px !important;}.vc_custom_1666103857641{margin-bottom: 15px !important;}.vc_custom_1468336959421{padding-right: 15px !important;}.vc_custom_1591979257539{margin-top: 10px !important;margin-right: 15px !important;background-color: #2864d7 !important;}.vc_custom_1558476788093{margin-top: 0px !important;margin-bottom: 15px !important;}</style><noscript><style type="text/css"> .wpb_animate_when_almost_visible { opacity: 1; }</style></noscript></head>
<body class="home page-template-default page page-id-682 header-full-width full-width-content show-util-bar multi-column-menu blue-footer show-subtitle show-sitetitle  no-post-title show-titlearea wpb-js-composer js-comp-ver-5.7 vc_responsive gray-background" itemscope itemtype="https://schema.org/WebPage">    <!-- Google Tag Manager (noscript) -->
    <noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-W2ZN7TC"
    height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
    <!-- End Google Tag Manager (noscript) -->
<div class="site-container"><!-- !.page-top -->
<section id="page-top">
        <!-- Rave Alert displays here #1 -->
    <div id="rave-alert" style="display: none; position: relative; z-index: 22;" role="region" aria-label="Emergency Information"></div>
<div id="mobile-utility-bar">
            <div class="wrap">
                <ul class="mobile-header-links clearfix">
                    <li class="vert-ellipsis">
                        <i class="fa fa-ellipsis-v" aria-hidden="true"></i>
                    </li>
                    <li class="header-links alumni-header-links menu-btn">
                        <a href="https://alumni.gsu.edu" target="_blank">Alumni</a>
                    </li>
                                        <li class="header-links staff-header-links mobile-menu-btn">
                        <a href="javascript:void(0);" title="Click for Faculty &amp; Staff Menu">Faculty &amp; Staff</a>
                                                <div class="triangle"></div>
                    </li>
                                        <li class="header-links student-header-links mobile-menu-btn">
                        <a href="javascript:void(0);" title="Click for Student Menu">Students</a>
                                                <div class="triangle"></div>
                    </li>
                </ul>
                <!-- mobile header-menu-area -->
                <div id="mobile-header-menu">
                        <div class="student-header-menu">
                                                    <div class="student-external-menu top-level"><span class="top-lvl-btn">Georgia State Menu</span>
                            <div class="student-staff-quick-links">
                                <div class="menu-network_student_header_menu-container"><ul id="menu-network_student_header_menu" class="menu"><li id="menu-item-4183" class=" menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4183"><a href="#">Academic Life</a><ul class="sub-menu"><li id="menu-item-4144" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4144"><a href="https://registrar.gsu.edu/registration/semester-calendars-exam-schedules/">Academic Calendars</a></li><li id="menu-item-4172" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4172"><a href="https://library.gsu.edu">Library</a></li><li id="menu-item-4143" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4143"><a href="https://catalogs.gsu.edu/">University Catalog</a></li><li id="menu-item-4137" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4137"><a href="https://paws.gsu.edu/">Class Registration</a></li><li id="menu-item-4138" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4138"><a href="https://registrar.gsu.edu/registration/registration-guide/">Registration Information</a></li><li id="menu-item-4139" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4139"><a href="https://advisement.gsu.edu/">Student Advisement</a></li><li id="menu-item-4181" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4181"><a href="https://graduate.gsu.edu/current-students/">Grad Student Resources</a></li><li id="menu-item-4140" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4140"><a href="https://success.students.gsu.edu/">Student Success</a></li><li id="menu-item-10531" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-10531"><a href="https://success.students.gsu.edu/learning-tutoring-center/">Learning & Tutoring Center</a></li><li id="menu-item-4141" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4141"><a href="https://veterans.gsu.edu/">Military Outreach</a></li><li id="menu-item-4142" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4142"><a href="https://advisement.gsu.edu/self-service/gpa-calculator/">GPA Calculator</a></li><li id="menu-item-4177" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4177"><a href="https://gsu.my.site.com/support/s/">Panther Answers</a></li></ul></li><li id="menu-item-4162" class="column-two menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4162"><a href="https://sfs.gsu.edu/">Financial Services</a><ul class="sub-menu"><li id="menu-item-4147" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4147"><a href="https://sfs.gsu.edu/tuition-fees/payments/">Tuition & Payments</a></li><li id="menu-item-4157" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4157"><a href="https://sfs.gsu.edu/tuition-fees/tuition-classification/">Tuition Classification</a></li><li id="menu-item-4145" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4145"><a href="https://sfs.gsu.edu/scholarships-grants/">Scholarship Information</a></li><li id="menu-item-4178" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4178"><a href="https://success.students.gsu.edu/scholarship-resource-center/">Search for Scholarships</a></li><li id="menu-item-4146" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4146"><a href="https://sfs.gsu.edu/the-financial-aid-process/">Financial Aid</a></li><li id="menu-item-4160" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4160"><a href="https://sfs.gsu.edu/loans-work-study/loans/">Loans</a></li><li id="menu-item-4148" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4148"><a href="https://sfs.gsu.edu/the-financial-aid-process/applying-for-aid/">FAFSA</a></li><li id="menu-item-4158" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4158"><a href="https://sfs.gsu.edu/tuition-fees/overpayment-refunds/">Refunds</a></li><li id="menu-item-4159" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4159"><a href="https://sfs.gsu.edu/tuition-fees/student-health-insurance/">Student Health Insurance</a></li><li id="menu-item-12883" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-12883"><a href="https://sfs.gsu.edu/resources/panther-book-access-program/">Panther Book Access Program</a></li><li id="menu-item-4161" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4161"><a href="https://sfs.gsu.edu/loans-work-study/student-employment/">Student Employment</a></li><li id="menu-item-4180" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4180"><a href="https://ombuds.gsu.edu/">Ombudsperson</a></li></ul></li><li id="menu-item-4150" class="column-three menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4150"><a href="https://engagement.gsu.edu/">Student Engagement</a><ul class="sub-menu"><li id="menu-item-5130" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-5130"><a href="https://engagement.gsu.edu/help/">Find Help</a></li><li id="menu-item-4182" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4182"><a href="https://pin.gsu.edu">Student Organizations</a></li><li id="menu-item-4163" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4163"><a href="https://engagement.gsu.edu/get-involved/programs/">Programs</a></li><li id="menu-item-4154" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4154"><a href="https://engagement.gsu.edu/explore/nurture-healthy-habits/">Health & Well-being</a></li><li id="menu-item-4156" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4156"><a href="https://recreation.gsu.edu/">Recreation</a></li><li id="menu-item-4168" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4168"><a href="https://spotlight.gsu.edu/">Spotlight Programs</a></li><li id="menu-item-4155" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4155"><a href="https://career.gsu.edu/">Career Services</a></li><li id="menu-item-4179" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4179"><a href="https://studentcenter.gsu.edu/">Student Center</a></li><li id="menu-item-4151" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4151"><a href="https://myhousing.gsu.edu/">Housing</a></li><li id="menu-item-4167" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4167"><a href="https://sga.gsu.edu/">Student Government</a></li><li id="menu-item-4164" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4164"><a href="https://studenthandbook.gsu.edu/">Student Handbook</a></li><li id="menu-item-4165" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4165"><a href="https://codeofconduct.gsu.edu/">Code of Conduct</a></li></ul></li><li id="menu-item-4170" class="column-four menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4170"><a href="https://technology.gsu.edu/guides/it-resources-for-students/">Technology</a><ul class="sub-menu"><li id="menu-item-4173" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4173"><a href="https://technology.gsu.edu/technology-services/email/panthermail-student-email/">Email</a></li><li id="menu-item-4174" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4174"><a href="https://paws.gsu.edu/">PAWS</a></li><li id="menu-item-4175" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4175"><a href="https://paws.gsu.edu/student-resources/">GoSOLAR</a></li><li id="menu-item-4176" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4176"><a href="https://icollege.gsu.edu/">iCollege</a></li><li id="menu-item-12241" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-12241"><a href="https://cetloe.gsu.edu/digital-learning-at-georgia-state/">Digital Learning@GSU</a></li></ul></li><li id="menu-item-4166" class="column-four menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4166"><a href="https://services.gsu.edu">Campus Services</a><ul class="sub-menu"><li id="menu-item-4169" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4169"><a href="https://www.getrave.com/login/gsu">Get Emergency Alerts</a></li><li id="menu-item-4152" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4152"><a href="https://parking.gsu.edu/">Parking & Transportation</a></li><li id="menu-item-4153" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4153"><a href="https://dining.gsu.edu/">PantherDining</a></li><li id="menu-item-4149" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4149"><a href="https://services.gsu.edu/university-bookstore/">Bookstore</a></li><li id="menu-item-4171" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4171"><a href="https://panthercard.gsu.edu/">PantherCard</a></li></ul></li></ul></div>                                <ul class="menu">
                                    <li class="extras"><span>Useful Links</span>
                                        <ul class="sub-menu">
                                            <li>
                                                 <a href="https://news.gsu.edu/coronavirus-news-hub/" target="_blank">Coronavirus News</a>
                                           </li>
                                            <li><a href="https://campusdirectory.gsu.edu/" target="_blank">Directory (Login Required)</a></li>
                                            <li><a href="https://www.gsu.edu/a-z-index/">Student A-Z Index</a></li>
                                            <li>
                                                 <a href="http://calendar.gsu.edu/" target="_blank"><div class="Calendar"></div></a>
                                           </li>
                                            <li>
                                                <a href="https://lawlibrary.gsu.edu/">Law Library</a>
                                            <li><a href="https://library.gsu.edu/">Library</a></li>
                                            </li>
                                            <li>
                                            <li><a href="https://gsutech.service-now.com/sp">Help Center</a></li>
                                           </li>
                                            <li><a href="https://safety.gsu.edu/">Safety &amp; Security</a></li>
                                            <li><a href="https://facilities.gsu.edu/">Facilities</a></li>
                                            <li><a href="https://audit.gsu.edu/georgia-state-ethics-hotline/">Ethics Hotline</a></li>
                                            <li><a href="https://gsu.starrezhousing.com/StarRezPortalX/B1834504/1/1/Home-Home?UrlToken=0D72A24E">MyHousing (STARREZ) Portal</a></li>
                                            <li><a href="https://safety.gsu.edu/livesafe/">Get The LiveSafe App</a></li>
                                        </ul>
                                    </li>
                                </ul>
                            </div>
                            </div>
                        </div>
                        <div class="mobile-student-widget-area">
                                                </div>
                        <div class="faculty-staff-header-menu">
                                                    <div class="staff-external-menu top-level"><span class="top-lvl-btn">Georgia State Menu</span>
                            <div class="student-staff-quick-links">
                                <div class="menu-network_staff_header_menu-container"><ul id="menu-network_staff_header_menu" class="menu"><li id="menu-item-4088" class="column-one menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4088"><a href="https://finance.gsu.edu">Finance & Admin.</a><ul class="sub-menu"><li id="menu-item-4089" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4089"><a href="https://finance.gsu.edu/tools/">Toolkit</a></li><li id="menu-item-4090" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4090"><a href="https://finance.gsu.edu/forms-library/">Forms</a></li><li id="menu-item-4091" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4091"><a href="https://finance.gsu.edu/training/">Training</a></li><li id="menu-item-4092" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4092"><a href="https://policies.oie.gsu.edu/">University Policies</a></li><li id="menu-item-4120" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4120"><a href="https://senate.gsu.edu/">University Senate</a></li><li id="menu-item-7665" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-7665"><a href="https://staffcouncil.gsu.edu">Staff Council</a></li><li id="menu-item-4094" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4094"><a href="https://finance.gsu.edu/budget-and-planning/">Budget & Planning</a></li><li id="menu-item-4093" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4093"><a href="https://finance.gsu.edu/#dapt">Disbursement & Accts. Payable</a></li><li id="menu-item-4096" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4096"><a href="https://finance.gsu.edu/purchasing-business/">Purchasing & Business</a></li><li id="menu-item-4095" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4095"><a href="https://finance.gsu.edu/spectrum-services/">Spectrum Services</a></li><li id="menu-item-4135" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4135"><a href="https://risk.gsu.edu">Risk Management</a></li><li id="menu-item-4133" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4133"><a href="https://ombuds.gsu.edu/">Ombudsperson</a></li></ul></li><li id="menu-item-4097" class="column-two menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4097"><a href="https://hr.gsu.edu">Human Resources</a><ul class="sub-menu"><li id="menu-item-4100" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4100"><a href="https://finance.gsu.edu/financial-services/payroll-time-and-absence/">Payroll & W2 Information</a></li><li id="menu-item-4098" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4098"><a href="https://hr.gsu.edu/service-centers/benefits/">Benefits</a></li><li id="menu-item-4099" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4099"><a href="https://hr.gsu.edu/benefits/time-off-and-time-away/">Vacation & Leave</a></li><li id="menu-item-4106" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4106"><a href="https://hr.gsu.edu/employee-success-center/">Employee Success Center</a></li><li id="menu-item-4105" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4105"><a href="https://hr.gsu.edu/development-engagement/new-employee-engagement/">New Hires</a></li><li id="menu-item-4107" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4107"><a href="https://hr.gsu.edu/employee-resources/">Employee Resources</a></li><li id="menu-item-4104" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4104"><a href="https://hr.gsu.edu/manager-resources/">Managers & HR Partners</a></li><li id="menu-item-4103" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4103"><a href="https://benefits.usg.edu/benefits-resources/planning-for-retirement">Retired or Planning to Retire</a></li><li id="menu-item-4101" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4101"><a href="https://faculty.gsu.edu/faculty-handbook/">Faculty Handbook</a></li><li id="menu-item-4102" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4102"><a href="https://hr.gsu.edu/staff-handbook">Staff Handbook</a></li></ul></li><li id="menu-item-4108" class="column-three menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4108"><a href="https://services.gsu.edu">Campus Services</a><ul class="sub-menu"><li id="menu-item-4109" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4109"><a href="https://facilities.gsu.edu/">Facilities Management</a></li><li id="menu-item-4110" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4110"><a href="https://ursa.research.gsu.edu/">URSA</a></li><li id="menu-item-4118" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4118"><a href="https://services.gsu.edu/service/mailservices/">Mail Services</a></li><li id="menu-item-4119" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4119"><a href="https://pawprints.gsu.edu">Printing</a></li><li id="menu-item-4112" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4112"><a href="https://dining.gsu.edu/">PantherDining & Catering</a></li><li id="menu-item-4113" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4113"><a href="https://parking.gsu.edu/">Parking & Transportation</a></li><li id="menu-item-4128" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4128"><a href="https://finance.gsu.edu/#travel">Travel Reservations</a></li><li id="menu-item-4114" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4114"><a href="https://pr.gsu.edu">PR & Marketing Communications</a></li><li id="menu-item-4116" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4116"><a href="https://legalaffairs.gsu.edu/legal-services-forms/">Legal Services</a></li><li id="menu-item-4117" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4117"><a href="https://oie.gsu.edu">Institutional Effectiveness</a></li><li id="menu-item-4111" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4111"><a href="https://emeriti.gsu.edu/">Emeriti Association</a></li></ul></li><li id="menu-item-4115" class="column-four menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4115"><a href="https://technology.gsu.edu/guides/it-resources-for-faculty-staff/">Technology</a><ul class="sub-menu"><li id="menu-item-4121" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4121"><a href="https://outlook.com/gsu.edu">Email</a></li><li id="menu-item-4122" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4122"><a href="https://sendafile.gsu.edu/">Send A File</a></li><li id="menu-item-12863" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-12863"><a href="https://faculty.gsu.edu/digitalmeasures/">Digital Measures</a></li><li id="menu-item-4123" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4123"><a href="https://paws.gsu.edu/">PAWS</a></li><li id="menu-item-4124" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4124"><a href="https://paws.gsu.edu/faculty-resources/">GoSOLAR</a></li><li id="menu-item-4125" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4125"><a href="https://icollege.gsu.edu/">iCollege</a></li><li id="menu-item-4130" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4130"><a href="https://oneusgconnect.usg.edu/">OneUSG Connect</a></li><li id="menu-item-4126" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4126"><a href="https://panthermart.gsu.edu">Panthermart</a></li><li id="menu-item-4127" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4127"><a href="https://www.spectrum.gsu.edu/psp/FPROD/?cmd=login&languageCd=ENG&">Spectrum (requires VPN)</a></li><li id="menu-item-12452" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-12452"><a href="https://www.gsu.edu/asset-works">Asset Works</a></li><li id="menu-item-5296" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-5296"><a href="https://technology.gsu.edu/technology-services/technology-professional-training/">Training and Learning</a></li><li id="menu-item-4129" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4129"><a href="https://cetl.gsu.edu/">CETLOE</a></li><li id="menu-item-4136" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4136"><a href="https://gsu.catalog.instructure.com/">Stacks for State</a></li></ul></li></ul></div>                                <ul class="menu">
                                    <li class="extras"><span>Useful Links</span>
                                        <ul class="sub-menu">
                                            <li>
                                                <a href="https://news.gsu.edu/coronavirus-news-hub/" target="_blank">Coronavirus News</a>
                                            </li>
                                            <li><a href="https://campusdirectory.gsu.edu/" target="_blank">Directory (Login Required)</a></li>
                                            <li><a href="https://www.gsu.edu/a-z-index/">Student A-Z Index</a></li>
                                            <li>
                                                 <a href="http://calendar.gsu.edu/" target="_blank"><div class="Calendar"></div></a>
                                            </li>
                                            <li>
                                            <a href="https://lawhelp.gsu.edu" target="_blank">Help Center</a>
                                            <li><a href="https://gsutech.service-now.com/sp">Help Center</a></li>
                                            </li>
                                            <li><a href="https://safety.gsu.edu/">Safety &amp; Security</a></li>
                                            <li><a href="https://facilities.gsu.edu/">Facilities</a></li>
                                            <li><a href="https://audit.gsu.edu/georgia-state-ethics-hotline/">Ethics Hotline</a></li>
                                            <li><a href="https://commkit.gsu.edu">Identity & Comms. ToolKit</a></li>
                                            <li><a href="https://www.gsu.edu/collegetocareer">College to Career</a></li>
                                            <li><a href="https://safety.gsu.edu/livesafe/">Get The LiveSafe App</a></li>
                                        </ul>
                                    </li>
                                </ul>
                            </div>
                            </div>
                        </div>
                        <div class="mobile-staff-widget-area">
                                                </div>
                </div>
                <!--  end mobile header-menu-area -->
                <ul>
                  <li class="navicon">
                        <!-- <i class="fa fa-bars" aria-hidden="true"></i> -->
                </li>

                      <li class="mobile-title "><a href="https://news.gsu.edu">News Hub</a></li>
                                                  <li class="breadcrumb header-menu-tab student network">Georgia State Menu</li>
                                        <li class="breadcrumb header-menu-tab staff network">Georgia State Menu</li>
                    <li class="vert-ellipsis">
                        <i class="fa fa-ellipsis-v" aria-hidden="true"></i>
                    </li>
                    <li class="search">
                        <div id="mobile-site-search">
                            <form action="//www.gsu.edu/search-georgia-state/" method="get" name="searchform" class="form-search searchform" id="searchform">
                                <!--  Remove Directory Search Fields
                                <input type=hidden value="*" name="givennames" />
                                <input type=hidden value="*" name="sns" />
                                <input type=hidden value="All" name="search" id="a1212" />
                                <input type=hidden value="" name="sn" id="sn" />
                                <input type=hidden value="" name="givenname" id="givenname" />
                                Remove Directory Search Radio -->
                                <div class="input-append">
                                    <div class="util-icon-search"></div><input class="input-medium" placeholder="SEARCH" type="text" value="" name="qm"  size="16"  id="qm" autocomplete="off" /><label for="qm" class="visually-hidden">Search: </label><button name="Submit" value="Submit" class="btn" type="submit"><span class="dashicons dashicons-arrow-right-alt"></span></button>
                                </div>
                                <div class="dropdown search-dropdown">
                                    <ul>
                                        <li>
                                            <input type="radio" name="site" value="GeorgiaState" id="radio-gsu-mobile" checked>
                                            <label class="radio" for="radio-gsu-mobile">Georgia State</label>
                                        </li>
                                <!--  Remove Directory Search Radio
                                        <li>
                                            <input type="radio" name="site" value="directory" id="radio-dir-mobile">
                                            <label class="radio" for="radio-dir-mobile">Campus Directory</label>
                                        </li>
                                Remove Directory Search Radio -->
                                                                            </ul>
                                </div>
                            </form>
                        </div>
                    </li>
                    <li class="campus-map">
                        <a aria-label="Campus Map" href="https://map.concept3d.com/?id=1108#!ct/0" target="_blank"><i class="icon-campus-map"></i></a>
                    </li>
                    <li class="calendar">
                      <a aria-label="Calendar" href="http://calendar.gsu.edu/" target="_blank"><div class="calendar-icon"></div></a>
                    </li>
                </ul>
                <ul class="mobile-breadcrumb">
                                        <li class="breadcrumb off-site-menu">
                        <a href="https://www.gsu.edu">Georgia State Home<i class="icon icon-chevron-down-small"></i></a>
                        <div class="dropdown">
                                                    </div>
                    </li>
                                                        </ul>
            </div>
        </div>
                    <h2 class="screen-reader-text">Main navigation</h2>
            <nav class="mobile-nav-primary" itemscope="" itemtype="https://schema.org/SiteNavigationElement" aria-label="Main navigation">
                   <div class="menu-top-navigation-container"><ul id="mobilenav" class="mobile-menu"><li id="menu-item-6523545" class="menu-item menu-item-type-custom menu-item-object-custom current-menu-item current_page_item menu-item-has-children menu-item-6523545 thirds"><a href="/">News<div class="sub-title"></div></a>
<ul class="sub-menu">
	<li id="menu-item-6523541" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-6523541 default-columns column-two"><a target="_blank" href="http://www.georgiastatesports.com/">Athletics<div class="sub-title"></div></a></li>
	<li id="menu-item-6577629" class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-6577629 default-columns column-two"><a href="https://news.gsu.edu/category/beyond-college-to-career/">Beyond College to Career<div class="sub-title"></div></a></li>
	<li id="menu-item-6523540" class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-6523540 default-columns column-two"><a href="https://news.gsu.edu/category/campus-news/">Campus News<div class="sub-title"></div></a></li>
	<li id="menu-item-6549909" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6549909 default-columns column-two"><a href="https://news.gsu.edu/campus-safety/">Campus Safety<div class="sub-title"></div></a></li>
	<li id="menu-item-6554050" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6554050 default-columns column-two"><a href="https://news.gsu.edu/international-initiatives-news/">International News<div class="sub-title"></div></a></li>
	<li id="menu-item-6523548" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523548 default-columns column-two"><a href="https://news.gsu.edu/research-news/">Research News<div class="sub-title"></div></a></li>
	<li id="menu-item-6523546" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-6523546 default-columns column-three"><a href="#">College and Schools<div class="sub-title"></div></a>
	<ul class="sub-menu">
		<li id="menu-item-6523549" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523549 default-columns"><a href="https://news.gsu.edu/aysps-news/">Andrew Young School News<div class="sub-title"></div></a></li>
		<li id="menu-item-6523550" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523550 default-columns"><a href="https://news.gsu.edu/arts-and-sciences-news/">College of Arts and Sciences News<div class="sub-title"></div></a></li>
		<li id="menu-item-6523554" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523554 default-columns"><a href="https://news.gsu.edu/college-arts-news/">College of the Arts News<div class="sub-title"></div></a></li>
		<li id="menu-item-6523551" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523551 default-columns"><a href="https://news.gsu.edu/cehd-news/">College of Education and Human Development News<div class="sub-title"></div></a></li>
		<li id="menu-item-6523552" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523552 default-columns"><a href="https://news.gsu.edu/law-news/">College of Law News<div class="sub-title"></div></a></li>
		<li id="menu-item-6523553" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523553 default-columns"><a href="https://news.gsu.edu/college-nursing-health-professions/">College of Nursing and Health Professions News<div class="sub-title"></div></a></li>
		<li id="menu-item-6523555" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523555 default-columns"><a href="https://news.gsu.edu/honors-news/">Honors College News<div class="sub-title"></div></a></li>
		<li id="menu-item-6523556" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523556 default-columns"><a href="https://news.gsu.edu/ibms-news/">Institute for Biomedical Sciences News<div class="sub-title"></div></a></li>
		<li id="menu-item-6523557" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523557 default-columns"><a href="https://news.gsu.edu/perimeter-news/">Perimeter College News<div class="sub-title"></div></a></li>
		<li id="menu-item-6560821" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6560821 default-columns"><a href="https://news.gsu.edu/robinson-news/">Robinson College of Business News<div class="sub-title"></div></a></li>
		<li id="menu-item-6523559" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523559 default-columns"><a href="https://news.gsu.edu/public-health-news/">School of Public Health News<div class="sub-title"></div></a></li>
	</ul>
</li>
	<li id="menu-item-6523560" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-6523560 default-columns column-one"><a href="#">News by Category<div class="sub-title"></div></a>
	<ul class="sub-menu">
		<li id="menu-item-3518" class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-3518 default-columns"><a href="https://news.gsu.edu/category/arts-culture/">Arts &#038; Culture<div class="sub-title"></div></a></li>
		<li id="menu-item-3520" class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-3520 default-columns"><a href="https://news.gsu.edu/category/business-economy/">Business &#038; Economy<div class="sub-title"></div></a></li>
		<li id="menu-item-3521" class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-3521 default-columns"><a href="https://news.gsu.edu/category/education-leadership/">Education &#038; Leadership<div class="sub-title"></div></a></li>
		<li id="menu-item-3523" class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-3523 default-columns"><a href="https://news.gsu.edu/category/health-wellness/">Health &#038; Wellness<div class="sub-title"></div></a></li>
		<li id="menu-item-3522" class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-3522 default-columns"><a href="https://news.gsu.edu/category/politics-law-society/">Politics, Law &#038; Society<div class="sub-title"></div></a></li>
		<li id="menu-item-3519" class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-3519 default-columns"><a href="https://news.gsu.edu/category/science-technology/">Science &#038; Technology<div class="sub-title"></div></a></li>
	</ul>
</li>
</ul>
</li>
<li id="menu-item-6523561" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-6523561 thirds"><a href="#">Magazines<div class="sub-title"></div></a>
<ul class="sub-menu">
	<li id="menu-item-6523562" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523562 default-columns"><a href="https://news.gsu.edu/magazine/">University Magazine<div class="sub-title"></div></a></li>
	<li id="menu-item-6527474" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-6527474 default-columns"><a href="https://news.gsu.edu/research-magazine">Research Magazine<div class="sub-title"></div></a></li>
</ul>
</li>
<li id="menu-item-6523542" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523542 default-columns"><a href="https://news.gsu.edu/press-kit/">Press<div class="sub-title"></div></a></li>
<li id="menu-item-6523539" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523539 default-columns"><a href="https://news.gsu.edu/experts/">Experts<div class="sub-title"></div></a></li>
<li id="menu-item-6523543" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-6523543 default-columns"><a target="_blank" href="https://calendar.gsu.edu">Calendar<div class="sub-title"></div></a></li>
</ul></div>                                                   <ul>
                    <li class="parent-home">
                        <a href="https://www.gsu.edu">Georgia State Home</a>
                                            </li>
                </ul>
                            </nav>
        <!-- end mobile util bar -->
      <div id="fixed-util-bar">
                <div id="utility-bar" role="navigation" aria-label="Utility Navigation">
            <div class="wrap">
                <ul>
          <li class="breadcrumb logo" tabindex="1"><a href="https://www.gsu.edu/" tabindex="-1"><img tabindex="-1" class="univmark" src="https://news.gsu.edu/wp-content/themes/gsu-flex-2.1-magazine/images/logo.png" alt="Geogia State University Logo"/></a></li>
                                      <li class="breadcrumb off-site-menu" tabindex="2">
                        <a href="https://www.gsu.edu">Georgia State Home<!--<i class="icon icon-chevron-down-small"></i>--></a>
                        <div class="dropdown">
                                                    </div>
                    </li>
                                                                                <li class="breadcrumb header-menu-tab student network" tabindex="-1">Georgia State Menu</li>
                                        <li class="breadcrumb header-menu-tab staff network" tabindex="-1">Georgia State Menu</li>
                    <li class="vert-ellipsis">
                        <i class="fa fa-ellipsis-v" aria-hidden="true"></i>
                    </li>
                    <li class="search">
                        <div id="site-search">
                            <form action="//www.gsu.edu/search-georgia-state/" method="get" name="searchform" class="form-search searchform" id="searchform">
                                <!-- Remove Directory Search Fields 
                                <input type=hidden value="*" name="givennames" />
                                <input type=hidden value="*" name="sns" />
                                <input type=hidden value="All" name="search" id="a1212" />
                                <input type=hidden value="" name="sn" id="sn" />
                                <input type=hidden value="" name="givenname" id="givenname" />
                                End Remove Directory Search Fields -->
                                <div class="input-append">
                                    <div class="util-icon-search"></div><label for="search" class="visually-hidden">Search: </label><input class="input-medium" placeholder="SEARCH" type="text" value="" name="q"  size="16"  id="q" autocomplete="off" /><label for="q" class="visually-hidden">Search: </label><button name="Submit" value="Submit" class="btn" type="submit"><span class="dashicons dashicons-arrow-right-alt"></span></button>
                                </div>
                                <div class="dropdown search-dropdown">
                                    <ul>
                                                                                <li>
                                            <input type="radio" name="site" value="GeorgiaState" id="radio-gsu" checked>
                                            <label class="radio" for="radio-gsu">Georgia State</label>
                                        </li>
                                <!-- Remove Directory Search Radio 
                                        <li>
                                            <input type="radio" name="site" value="directory" id="radio-dir">
                                            <label class="radio" for="radio-dir">Campus Directory</label>
                                        </li>
                                End Remove Directory Search Fields -->
                                                                            </ul>
                                </div>
                            </form>
                        </div>
                    </li>
                    <li class="campus-map" tabindex="7">
                        <a aria-label="Campus Map" href="https://map.concept3d.com/?id=1108#!ct/0" target="_blank" tabindex="-1"><i class="icon-campus-map" tabindex="-1"></i></a>
                    </li>
                    <li class="calendar" tabindex="6">
                        <a aria-label="Calendar" href="http://calendar.gsu.edu/" target="_blank" tabindex="-1"><i class="calendar-icon"></i></a>
                    </li>
                    <li class="header-links alumni-header-links menu-btn" tabindex="5">
                        <a href="https://alumni.gsu.edu" target="_blank">Alumni</a>
                    </li>
                    <li class="header-links staff-header-links menu-btn" tabindex="4">
                        <a href="javascript:void(0);" title="Click for Faculty &amp; Staff Menu" tabindex="-1">
                            Faculty &amp; Staff</a>
                                                <div class="triangle"></div>
                    </li>
                    <li class="header-links student-header-links menu-btn" tabindex="3">
                        <a href="javascript:void(0);" title="Click for Student Menu" tabindex="-1">
                            Students</a>
                                                <div class="triangle"></div>
                    </li>
                </ul>
            </div>
            <!-- header-menu-area -->
            <div id="header-menu" role="navigation" aria-label="Site Navigation">
                <div class="header-menu-container">
          <div class="student-header-menu">
                      <div class="student-staff-quick-links" role="navigation" aria-label="Student Links">
              <div class="menu-network_student_header_menu-container"><ul id="menu-network_student_header_menu" class="menu"><li id="menu-item-4183" class=" menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4183"><a href="#">Academic Life</a><ul class="sub-menu"><li id="menu-item-4144" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4144"><a href="https://registrar.gsu.edu/registration/semester-calendars-exam-schedules/">Academic Calendars</a></li><li id="menu-item-4172" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4172"><a href="https://library.gsu.edu">Library</a></li><li id="menu-item-4143" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4143"><a href="https://catalogs.gsu.edu/">University Catalog</a></li><li id="menu-item-4137" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4137"><a href="https://paws.gsu.edu/">Class Registration</a></li><li id="menu-item-4138" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4138"><a href="https://registrar.gsu.edu/registration/registration-guide/">Registration Information</a></li><li id="menu-item-4139" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4139"><a href="https://advisement.gsu.edu/">Student Advisement</a></li><li id="menu-item-4181" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4181"><a href="https://graduate.gsu.edu/current-students/">Grad Student Resources</a></li><li id="menu-item-4140" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4140"><a href="https://success.students.gsu.edu/">Student Success</a></li><li id="menu-item-10531" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-10531"><a href="https://success.students.gsu.edu/learning-tutoring-center/">Learning & Tutoring Center</a></li><li id="menu-item-4141" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4141"><a href="https://veterans.gsu.edu/">Military Outreach</a></li><li id="menu-item-4142" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4142"><a href="https://advisement.gsu.edu/self-service/gpa-calculator/">GPA Calculator</a></li><li id="menu-item-4177" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4177"><a href="https://gsu.my.site.com/support/s/">Panther Answers</a></li></ul></li><li id="menu-item-4162" class="column-two menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4162"><a href="https://sfs.gsu.edu/">Financial Services</a><ul class="sub-menu"><li id="menu-item-4147" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4147"><a href="https://sfs.gsu.edu/tuition-fees/payments/">Tuition & Payments</a></li><li id="menu-item-4157" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4157"><a href="https://sfs.gsu.edu/tuition-fees/tuition-classification/">Tuition Classification</a></li><li id="menu-item-4145" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4145"><a href="https://sfs.gsu.edu/scholarships-grants/">Scholarship Information</a></li><li id="menu-item-4178" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4178"><a href="https://success.students.gsu.edu/scholarship-resource-center/">Search for Scholarships</a></li><li id="menu-item-4146" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4146"><a href="https://sfs.gsu.edu/the-financial-aid-process/">Financial Aid</a></li><li id="menu-item-4160" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4160"><a href="https://sfs.gsu.edu/loans-work-study/loans/">Loans</a></li><li id="menu-item-4148" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4148"><a href="https://sfs.gsu.edu/the-financial-aid-process/applying-for-aid/">FAFSA</a></li><li id="menu-item-4158" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4158"><a href="https://sfs.gsu.edu/tuition-fees/overpayment-refunds/">Refunds</a></li><li id="menu-item-4159" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4159"><a href="https://sfs.gsu.edu/tuition-fees/student-health-insurance/">Student Health Insurance</a></li><li id="menu-item-12883" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-12883"><a href="https://sfs.gsu.edu/resources/panther-book-access-program/">Panther Book Access Program</a></li><li id="menu-item-4161" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4161"><a href="https://sfs.gsu.edu/loans-work-study/student-employment/">Student Employment</a></li><li id="menu-item-4180" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4180"><a href="https://ombuds.gsu.edu/">Ombudsperson</a></li></ul></li><li id="menu-item-4150" class="column-three menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4150"><a href="https://engagement.gsu.edu/">Student Engagement</a><ul class="sub-menu"><li id="menu-item-5130" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-5130"><a href="https://engagement.gsu.edu/help/">Find Help</a></li><li id="menu-item-4182" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4182"><a href="https://pin.gsu.edu">Student Organizations</a></li><li id="menu-item-4163" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4163"><a href="https://engagement.gsu.edu/get-involved/programs/">Programs</a></li><li id="menu-item-4154" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4154"><a href="https://engagement.gsu.edu/explore/nurture-healthy-habits/">Health & Well-being</a></li><li id="menu-item-4156" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4156"><a href="https://recreation.gsu.edu/">Recreation</a></li><li id="menu-item-4168" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4168"><a href="https://spotlight.gsu.edu/">Spotlight Programs</a></li><li id="menu-item-4155" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4155"><a href="https://career.gsu.edu/">Career Services</a></li><li id="menu-item-4179" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4179"><a href="https://studentcenter.gsu.edu/">Student Center</a></li><li id="menu-item-4151" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4151"><a href="https://myhousing.gsu.edu/">Housing</a></li><li id="menu-item-4167" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4167"><a href="https://sga.gsu.edu/">Student Government</a></li><li id="menu-item-4164" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4164"><a href="https://studenthandbook.gsu.edu/">Student Handbook</a></li><li id="menu-item-4165" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4165"><a href="https://codeofconduct.gsu.edu/">Code of Conduct</a></li></ul></li><li id="menu-item-4170" class="column-four menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4170"><a href="https://technology.gsu.edu/guides/it-resources-for-students/">Technology</a><ul class="sub-menu"><li id="menu-item-4173" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4173"><a href="https://technology.gsu.edu/technology-services/email/panthermail-student-email/">Email</a></li><li id="menu-item-4174" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4174"><a href="https://paws.gsu.edu/">PAWS</a></li><li id="menu-item-4175" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4175"><a href="https://paws.gsu.edu/student-resources/">GoSOLAR</a></li><li id="menu-item-4176" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4176"><a href="https://icollege.gsu.edu/">iCollege</a></li><li id="menu-item-12241" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-12241"><a href="https://cetloe.gsu.edu/digital-learning-at-georgia-state/">Digital Learning@GSU</a></li></ul></li><li id="menu-item-4166" class="column-four menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4166"><a href="https://services.gsu.edu">Campus Services</a><ul class="sub-menu"><li id="menu-item-4169" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4169"><a href="https://www.getrave.com/login/gsu">Get Emergency Alerts</a></li><li id="menu-item-4152" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4152"><a href="https://parking.gsu.edu/">Parking & Transportation</a></li><li id="menu-item-4153" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4153"><a href="https://dining.gsu.edu/">PantherDining</a></li><li id="menu-item-4149" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4149"><a href="https://services.gsu.edu/university-bookstore/">Bookstore</a></li><li id="menu-item-4171" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4171"><a href="https://panthercard.gsu.edu/">PantherCard</a></li></ul></li></ul></div>                            <li class="extras header-close student-close"><a href="javascript:void(0);" title="Close Menu">&times;</a></li>
                            <li class="extras"><span>Useful Links</span>
                                <ul class="sub-menu">
                                      <li>                                  
                                                <a href="https://news.gsu.edu/coronavirus-news-hub/" target="_blank">Coronavirus News</a>
                                    </li>
                                    <li><a href="https://campusdirectory.gsu.edu/" target="_blank">Directory (Login Required)</a></li>
                                    <li><a href="https://www.gsu.edu/a-z-index/">Staff A-Z Index</a></li>
                                    <li>
                                    <a href="http://calendar.gsu.edu/" target="_blank">Calendar</a>
                                    </li>
                                     <li><a href="https://technology.gsu.edu/help-center/">Help Center</a></li>
                                    </li>
                                    <li><a href="https://safety.gsu.edu/">Safety &amp; Security</a></li>
                                    <li><a href="https://facilities.gsu.edu/">Facilities</a></li>
                                    <li><a href="https://audit.gsu.edu/georgia-state-ethics-hotline/">Ethics Hotline</a></li>
                                    <li><a href="https://www.gsu.edu/collegetocareer">College to Career</a></li>
                                    <li><a href="https://gsu.starrezhousing.com/StarRezPortalX/B1834504/1/1/Home-Home?UrlToken=0D72A24E">MyHousing (STARREZ) Portal</a></li>
                                    <li><a href="https://safety.gsu.edu/livesafe/">Get The LiveSafe App</a></li>
                                </ul>
                            </li>
            </div>

                                  </div>
      <div class="alumni-header-menu">
                    <div class="faculty-staff-header-menu" role="navigation" aria-label="Faculty and Staff Links">
                      <div class="student-staff-quick-links">
              <div class="menu-network_staff_header_menu-container"><ul id="menu-network_staff_header_menu" class="menu"><li id="menu-item-4088" class="column-one menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4088"><a href="https://finance.gsu.edu">Finance & Admin.</a><ul class="sub-menu"><li id="menu-item-4089" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4089"><a href="https://finance.gsu.edu/tools/">Toolkit</a></li><li id="menu-item-4090" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4090"><a href="https://finance.gsu.edu/forms-library/">Forms</a></li><li id="menu-item-4091" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4091"><a href="https://finance.gsu.edu/training/">Training</a></li><li id="menu-item-4092" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4092"><a href="https://policies.oie.gsu.edu/">University Policies</a></li><li id="menu-item-4120" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4120"><a href="https://senate.gsu.edu/">University Senate</a></li><li id="menu-item-7665" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-7665"><a href="https://staffcouncil.gsu.edu">Staff Council</a></li><li id="menu-item-4094" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4094"><a href="https://finance.gsu.edu/budget-and-planning/">Budget & Planning</a></li><li id="menu-item-4093" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4093"><a href="https://finance.gsu.edu/#dapt">Disbursement & Accts. Payable</a></li><li id="menu-item-4096" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4096"><a href="https://finance.gsu.edu/purchasing-business/">Purchasing & Business</a></li><li id="menu-item-4095" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4095"><a href="https://finance.gsu.edu/spectrum-services/">Spectrum Services</a></li><li id="menu-item-4135" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4135"><a href="https://risk.gsu.edu">Risk Management</a></li><li id="menu-item-4133" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4133"><a href="https://ombuds.gsu.edu/">Ombudsperson</a></li></ul></li><li id="menu-item-4097" class="column-two menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4097"><a href="https://hr.gsu.edu">Human Resources</a><ul class="sub-menu"><li id="menu-item-4100" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4100"><a href="https://finance.gsu.edu/financial-services/payroll-time-and-absence/">Payroll & W2 Information</a></li><li id="menu-item-4098" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4098"><a href="https://hr.gsu.edu/service-centers/benefits/">Benefits</a></li><li id="menu-item-4099" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4099"><a href="https://hr.gsu.edu/benefits/time-off-and-time-away/">Vacation & Leave</a></li><li id="menu-item-4106" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4106"><a href="https://hr.gsu.edu/employee-success-center/">Employee Success Center</a></li><li id="menu-item-4105" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4105"><a href="https://hr.gsu.edu/development-engagement/new-employee-engagement/">New Hires</a></li><li id="menu-item-4107" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4107"><a href="https://hr.gsu.edu/employee-resources/">Employee Resources</a></li><li id="menu-item-4104" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4104"><a href="https://hr.gsu.edu/manager-resources/">Managers & HR Partners</a></li><li id="menu-item-4103" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4103"><a href="https://benefits.usg.edu/benefits-resources/planning-for-retirement">Retired or Planning to Retire</a></li><li id="menu-item-4101" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4101"><a href="https://faculty.gsu.edu/faculty-handbook/">Faculty Handbook</a></li><li id="menu-item-4102" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4102"><a href="https://hr.gsu.edu/staff-handbook">Staff Handbook</a></li></ul></li><li id="menu-item-4108" class="column-three menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4108"><a href="https://services.gsu.edu">Campus Services</a><ul class="sub-menu"><li id="menu-item-4109" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4109"><a href="https://facilities.gsu.edu/">Facilities Management</a></li><li id="menu-item-4110" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4110"><a href="https://ursa.research.gsu.edu/">URSA</a></li><li id="menu-item-4118" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4118"><a href="https://services.gsu.edu/service/mailservices/">Mail Services</a></li><li id="menu-item-4119" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4119"><a href="https://pawprints.gsu.edu">Printing</a></li><li id="menu-item-4112" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4112"><a href="https://dining.gsu.edu/">PantherDining & Catering</a></li><li id="menu-item-4113" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4113"><a href="https://parking.gsu.edu/">Parking & Transportation</a></li><li id="menu-item-4128" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4128"><a href="https://finance.gsu.edu/#travel">Travel Reservations</a></li><li id="menu-item-4114" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4114"><a href="https://pr.gsu.edu">PR & Marketing Communications</a></li><li id="menu-item-4116" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4116"><a href="https://legalaffairs.gsu.edu/legal-services-forms/">Legal Services</a></li><li id="menu-item-4117" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4117"><a href="https://oie.gsu.edu">Institutional Effectiveness</a></li><li id="menu-item-4111" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4111"><a href="https://emeriti.gsu.edu/">Emeriti Association</a></li></ul></li><li id="menu-item-4115" class="column-four menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4115"><a href="https://technology.gsu.edu/guides/it-resources-for-faculty-staff/">Technology</a><ul class="sub-menu"><li id="menu-item-4121" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4121"><a href="https://outlook.com/gsu.edu">Email</a></li><li id="menu-item-4122" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4122"><a href="https://sendafile.gsu.edu/">Send A File</a></li><li id="menu-item-12863" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-12863"><a href="https://faculty.gsu.edu/digitalmeasures/">Digital Measures</a></li><li id="menu-item-4123" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4123"><a href="https://paws.gsu.edu/">PAWS</a></li><li id="menu-item-4124" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4124"><a href="https://paws.gsu.edu/faculty-resources/">GoSOLAR</a></li><li id="menu-item-4125" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4125"><a href="https://icollege.gsu.edu/">iCollege</a></li><li id="menu-item-4130" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4130"><a href="https://oneusgconnect.usg.edu/">OneUSG Connect</a></li><li id="menu-item-4126" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4126"><a href="https://panthermart.gsu.edu">Panthermart</a></li><li id="menu-item-4127" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4127"><a href="https://www.spectrum.gsu.edu/psp/FPROD/?cmd=login&languageCd=ENG&">Spectrum (requires VPN)</a></li><li id="menu-item-12452" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-12452"><a href="https://www.gsu.edu/asset-works">Asset Works</a></li><li id="menu-item-5296" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-5296"><a href="https://technology.gsu.edu/technology-services/technology-professional-training/">Training and Learning</a></li><li id="menu-item-4129" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4129"><a href="https://cetl.gsu.edu/">CETLOE</a></li><li id="menu-item-4136" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4136"><a href="https://gsu.catalog.instructure.com/">Stacks for State</a></li></ul></li></ul></div>                            <li class="extras header-close staff-close"><a href="javascript:void(0);" title="Close Menu">&times;</a></li>
                            <li class="extras"><span>Useful Links</span>
                                <ul class="sub-menu">
                                      <li>
                                            <a href="https://news.gsu.edu/coronavirus-news-hub/" target="_blank">Coronavirus News</a>
                                     </li>
                                    <li><a href="https://campusdirectory.gsu.edu/" target="_blank">Directory (Login Required)</a></li>
                                    <li><a href="https://www.gsu.edu/a-z-index/">Staff A-Z Index</a></li>
                                    <li>                                    <a href="https://news.gsu.edu/news-from-georgia-state" target="_blank">News & Updates</a>
                                    </li>
                                    <li>                                    <a href="https://library.gsu.edu/">Library</a>
                                    </li>
                                    <li>                                    <li><a href="https://technology.gsu.edu/help-center/">Help Center</a></li>
                                    </li>
                                    <li><a href="https://safety.gsu.edu/">Safety &amp; Security</a></li>
                                    <li><a href="https://facilities.gsu.edu/">Facilities</a></li>
                                    <li><a href="https://audit.gsu.edu/georgia-state-ethics-hotline/">Ethics Hotline</a></li>
                                    <li><a href="https://commkit.gsu.edu">Identity & Comms. ToolKit</a></li>
                                    <li><a href="https://www.gsu.edu/collegetocareer">College to Career</a></li>
                                    <li><a href="https://safety.gsu.edu/livesafe/">Get The LiveSafe App</a></li>
                                </ul>
                            </li>
            </div>
                                            </div>
                </div>
      </div>
      </div>
    </div>
    </section>
<ul class="genesis-skip-link"><li><a href="#genesis-content" class="screen-reader-shortcut"> Skip to content</a></li><li><a href="#genesis-nav-primary" class="screen-reader-shortcut"> Skip to primary nav</a></li></ul><header class="site-header" itemscope itemtype="https://schema.org/WPHeader"><div class="title-area"><h3 class="site-title" itemprop="headline"><a href="https://news.gsu.edu/">Georgia State News Hub</a></h3></div><span class="primary-nav-ellipsis vert-ellipsis"><i class="fa fa-ellipsis-v" aria-hidden="true"></i></span><h2 class="screen-reader-text">Main navigation</h2><nav class="nav-primary" itemscope="" itemtype="http://schema.org/SiteNavigationElement" id="genesis-nav-primary" aria-label="Main navigation"><ul id="menu-top-navigation" class="genesis-nav-menu"><li class="menu-item menu-item-type-custom menu-item-object-custom current-menu-item current_page_item menu-item-has-children menu-item-6523545 thirds"><a href="/">News<div class="sub-title"></div></a>
<ul class="sub-menu">
	<li class="menu-item menu-item-type-custom menu-item-object-custom menu-item-6523541 default-columns column-two"><a target="_blank" href="http://www.georgiastatesports.com/">Athletics<div class="sub-title"></div></a></li>
	<li class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-6577629 default-columns column-two"><a href="https://news.gsu.edu/category/beyond-college-to-career/">Beyond College to Career<div class="sub-title"></div></a></li>
	<li class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-6523540 default-columns column-two"><a href="https://news.gsu.edu/category/campus-news/">Campus News<div class="sub-title"></div></a></li>
	<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6549909 default-columns column-two"><a href="https://news.gsu.edu/campus-safety/">Campus Safety<div class="sub-title"></div></a></li>
	<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6554050 default-columns column-two"><a href="https://news.gsu.edu/international-initiatives-news/">International News<div class="sub-title"></div></a></li>
	<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523548 default-columns column-two"><a href="https://news.gsu.edu/research-news/">Research News<div class="sub-title"></div></a></li>
	<li class="menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-6523546 default-columns column-three"><a href="#">College and Schools<div class="sub-title"></div></a>
	<ul class="sub-menu">
		<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523549 default-columns"><a href="https://news.gsu.edu/aysps-news/">Andrew Young School News<div class="sub-title"></div></a></li>
		<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523550 default-columns"><a href="https://news.gsu.edu/arts-and-sciences-news/">College of Arts and Sciences News<div class="sub-title"></div></a></li>
		<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523554 default-columns"><a href="https://news.gsu.edu/college-arts-news/">College of the Arts News<div class="sub-title"></div></a></li>
		<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523551 default-columns"><a href="https://news.gsu.edu/cehd-news/">College of Education and Human Development News<div class="sub-title"></div></a></li>
		<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523552 default-columns"><a href="https://news.gsu.edu/law-news/">College of Law News<div class="sub-title"></div></a></li>
		<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523553 default-columns"><a href="https://news.gsu.edu/college-nursing-health-professions/">College of Nursing and Health Professions News<div class="sub-title"></div></a></li>
		<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523555 default-columns"><a href="https://news.gsu.edu/honors-news/">Honors College News<div class="sub-title"></div></a></li>
		<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523556 default-columns"><a href="https://news.gsu.edu/ibms-news/">Institute for Biomedical Sciences News<div class="sub-title"></div></a></li>
		<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523557 default-columns"><a href="https://news.gsu.edu/perimeter-news/">Perimeter College News<div class="sub-title"></div></a></li>
		<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6560821 default-columns"><a href="https://news.gsu.edu/robinson-news/">Robinson College of Business News<div class="sub-title"></div></a></li>
		<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523559 default-columns"><a href="https://news.gsu.edu/public-health-news/">School of Public Health News<div class="sub-title"></div></a></li>
	</ul>
</li>
	<li class="menu-item menu-item-type-custom menu-item-object-custom menu-item-6523560 default-columns column-one"><a href="#">News by Category<div class="sub-title"></div></a>
	<ul class="sub-menu">
		<li class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-3518 default-columns"><a href="https://news.gsu.edu/category/arts-culture/">Arts &#038; Culture<div class="sub-title"></div></a></li>
		<li class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-3520 default-columns"><a href="https://news.gsu.edu/category/business-economy/">Business &#038; Economy<div class="sub-title"></div></a></li>
		<li class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-3521 default-columns"><a href="https://news.gsu.edu/category/education-leadership/">Education &#038; Leadership<div class="sub-title"></div></a></li>
		<li class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-3523 default-columns"><a href="https://news.gsu.edu/category/health-wellness/">Health &#038; Wellness<div class="sub-title"></div></a></li>
		<li class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-3522 default-columns"><a href="https://news.gsu.edu/category/politics-law-society/">Politics, Law &#038; Society<div class="sub-title"></div></a></li>
		<li class="menu-item menu-item-type-taxonomy menu-item-object-category menu-item-3519 default-columns"><a href="https://news.gsu.edu/category/science-technology/">Science &#038; Technology<div class="sub-title"></div></a></li>
	</ul>
</li>
</ul>
</li>
<li class="menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-6523561 thirds"><a href="#">Magazines<div class="sub-title"></div></a>
<ul class="sub-menu">
	<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523562 default-columns"><a href="https://news.gsu.edu/magazine/">University Magazine<div class="sub-title"></div></a></li>
	<li class="menu-item menu-item-type-custom menu-item-object-custom menu-item-6527474 default-columns"><a href="https://news.gsu.edu/research-magazine">Research Magazine<div class="sub-title"></div></a></li>
</ul>
</li>
<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523542 default-columns"><a href="https://news.gsu.edu/press-kit/">Press<div class="sub-title"></div></a></li>
<li class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6523539 default-columns"><a href="https://news.gsu.edu/experts/">Experts<div class="sub-title"></div></a></li>
<li class="menu-item menu-item-type-custom menu-item-object-custom menu-item-6523543 default-columns"><a target="_blank" href="https://calendar.gsu.edu">Calendar<div class="sub-title"></div></a></li>
</ul></nav></header><div class="site-inner"><div class="content-sidebar-wrap"><main class="content" id="genesis-content"><article class="post-682 page type-page status-publish entry" itemscope itemtype="https://schema.org/CreativeWork"><header class="entry-header"><h1 class="screen-reader-text">Georgia State News</h1><h2 class="entry-title"><a href="https://news.gsu.edu/">Georgia State News</a></h2>
</header><div class="entry-content" itemprop="text"><div class="pf-content"><div id="news-feature" class="vc_row wpb_row vc_row-fluid vc_custom_1647455230012"><div class="shadow-box gradient wpb_column vc_column_container vc_col-sm-12 vc_hidden-sm vc_hidden-xs"><div class="vc_column-inner"><div class="wpb_wrapper"><!-- vc_grid start -->
<div class="vc_grid-container-wrapper vc_clearfix" >
	<div class="vc_grid-container vc_clearfix wpb_content_element vc_basic_grid news-feature" data-initial-loading-animation="none" data-vc-grid-settings="{&quot;page_id&quot;:682,&quot;style&quot;:&quot;all&quot;,&quot;action&quot;:&quot;vc_get_vc_grid_data&quot;,&quot;shortcode_id&quot;:&quot;1777651571214-ed55059c-3b97-9&quot;,&quot;tag&quot;:&quot;vc_basic_grid&quot;}" data-vc-request="https://news.gsu.edu/wp-admin/admin-ajax.php" data-vc-post-id="682" data-vc-public-nonce="bc96b7788b">
		<style type="text/css" data-type="vc_shortcodes-custom-css">.vc_custom_1469024685651{padding-right: 15px !important;}.vc_custom_1468286471174{padding-right: 15px !important;}.vc_custom_1475635317911{margin-top: 15px !important;}.vc_custom_1468286599423{padding-right: 15px !important;}.vc_custom_1468445804226{margin-top: 10px !important;margin-right: 10px !important;margin-bottom: 10px !important;margin-left: 10px !important;}.vc_custom_1467237032201{padding-right: 5px !important;padding-bottom: 15px !important;padding-left: 5px !important;}</style><div class="vc_grid vc_row vc_pageable-wrapper vc_hook_hover" data-vc-pageable-content="true"><div class="vc_pageable-slide-wrapper vc_clearfix" data-vc-grid-content="true"><div class="vc_grid-item vc_clearfix news-feature vc_col-sm-12 vc_grid-item-zone-c-left"><div class="vc_grid-item-mini vc_clearfix "><div class="vc_gitem-zone vc_gitem-zone-c vc_custom_1469024685651">
	<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align- vc_custom_1468286471174"><div class="vc_gitem-post-data feature-meta vc_gitem-post-data-source-post_categories vc_grid-filter vc_clearfix vc_grid-filter-comma  vc_grid-filter-size-md vc_grid-filter-center vc_grid-filter-color-grey"><div class="vc_grid-filter-item vc_gitem-post-category-name"><span class="vc_gitem-post-category-name"><a href="https://news.gsu.edu/category/campus-news/" alt="View all posts in Campus News" class="vc_gitem-link">Campus News</a></span></div>, <div class="vc_grid-filter-item vc_gitem-post-category-name"><span class="vc_gitem-post-category-name"><a href="https://news.gsu.edu/category/magazine/" alt="View all posts in Georgia State Magazine" class="vc_gitem-link">Georgia State Magazine</a></span></div></div>
<div class="vc_custom_heading h3-light-container vc_custom_1475635317911 vc_gitem-post-data vc_gitem-post-data-source-post_title" ><h3 style="text-align: left" ><a href="https://news.gsu.edu/magazine/the-golden-opportunity/" class="vc_gitem-link" title="The Golden Opportunity">The Golden Opportunity</a></h3></div><div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_excerpt" ><div style="text-align: left" ><a href="https://news.gsu.edu/magazine/the-golden-opportunity/" class="vc_gitem-link" title="The Golden Opportunity"><p>Thirty years after Atlanta hosted the 1996 Centennial Olympic Games, Georgia State looks back in a two-part series at how that moment reshaped both the city and the university for generations to come.</p>
</a></div></div><div class="vc_custom_heading feature-meta vc_gitem-post-data vc_gitem-post-data-source-post_date" ><div style="text-align: left" >April 14, 2026</div></div></div></div>	</div>
</div>
<div class="vc_gitem-animated-block  vc_gitem-animate vc_gitem-animate-none" data-vc-animation="none"><div class="vc_gitem-zone vc_gitem-zone-a vc_custom_1468286599423 vc-gitem-zone-height-mode-auto vc-gitem-zone-height-mode-auto-16-9" style="background-image: url('https://news.gsu.edu/files/2026/04/Stadium-fireworks-top-image-1200x800.jpg') !important;">
		<img src="https://news.gsu.edu/files/2026/04/Stadium-fireworks-top-image-1200x800.jpg" class="vc_gitem-zone-img" alt="">	<div class="vc_gitem-zone-mini">
			</div>
</div>
<div class="vc_gitem-zone vc_gitem-zone-b vc_custom_1468445804226 vc-gitem-zone-height-mode-auto">
			<div class="vc_gitem-zone-mini">
			</div>
</div>
</div>
</div><div class="vc_clearfix"></div></div></div></div>	</div>
</div><!-- vc_grid end -->
</div></div></div></div><div class="vc_row wpb_row vc_row-fluid feature-box vc_custom_1468846886472 vc_row-has-fill"><div class="shadow-box gradient wpb_column vc_column_container vc_col-sm-12 vc_hidden-lg vc_hidden-md"><div class="vc_column-inner"><div class="wpb_wrapper"><!-- vc_grid start -->
<div class="vc_grid-container-wrapper vc_clearfix" >
	<div class="vc_grid-container vc_clearfix wpb_content_element vc_basic_grid mobile-news-feature" data-initial-loading-animation="none" data-vc-grid-settings="{&quot;page_id&quot;:682,&quot;style&quot;:&quot;all&quot;,&quot;action&quot;:&quot;vc_get_vc_grid_data&quot;,&quot;shortcode_id&quot;:&quot;1777651571215-9e30d366-f649-2&quot;,&quot;tag&quot;:&quot;vc_basic_grid&quot;}" data-vc-request="https://news.gsu.edu/wp-admin/admin-ajax.php" data-vc-post-id="682" data-vc-public-nonce="bc96b7788b">
		<div class="vc_grid vc_row vc_pageable-wrapper vc_hook_hover" data-vc-pageable-content="true"><div class="vc_pageable-slide-wrapper vc_clearfix" data-vc-grid-content="true"><div class="vc_grid-item vc_clearfix vc_col-sm-12 vc_grid-item-zone-c-bottom"><div class="vc_grid-item-mini vc_clearfix "><div class="vc_gitem-animated-block "></div>
<div class="vc_gitem-zone vc_gitem-zone-c">
	<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-"><div class="vc_gitem-post-data vc_gitem-post-data-source-post_categories vc_grid-filter vc_clearfix vc_grid-filter-comma vc_grid-filter-size-md vc_grid-filter-center vc_grid-filter-color-grey"><div class="vc_grid-filter-item vc_gitem-post-category-name"><span class="vc_gitem-post-category-name">Campus News</span></div><div class="vc_grid-filter-item vc_gitem-post-category-name"><span class="vc_gitem-post-category-name">Georgia State Magazine</span></div></div>
<div class="vc_custom_heading h3-light-container vc_gitem-post-data vc_gitem-post-data-source-post_title" ><h3 style="text-align: left" ><a href="https://news.gsu.edu/magazine/the-golden-opportunity/" class="vc_gitem-link" title="The Golden Opportunity">The Golden Opportunity</a></h3></div>
	<div class="wpb_single_image wpb_content_element vc_align_center">
		<figure class="wpb_wrapper vc_figure">
			<a href="https://news.gsu.edu/magazine/the-golden-opportunity/" class="vc_gitem-link vc_single_image-wrapper vc_box_border_grey" title="The Golden Opportunity"><img width="1200" height="800" src="https://news.gsu.edu/files/2026/04/Stadium-fireworks-top-image-1200x800.jpg" class="vc_single_image-img attachment-full" alt="" srcset="https://news.gsu.edu/files/2026/04/Stadium-fireworks-top-image-1200x800.jpg 1200w, https://news.gsu.edu/files/2026/04/Stadium-fireworks-top-image-1200x800-300x200.jpg 300w, https://news.gsu.edu/files/2026/04/Stadium-fireworks-top-image-1200x800-1080x720.jpg 1080w, https://news.gsu.edu/files/2026/04/Stadium-fireworks-top-image-1200x800-768x512.jpg 768w, https://news.gsu.edu/files/2026/04/Stadium-fireworks-top-image-1200x800-450x300.jpg 450w, https://news.gsu.edu/files/2026/04/Stadium-fireworks-top-image-1200x800-900x600.jpg 900w" sizes="(max-width: 1200px) 100vw, 1200px" /></a>
		</figure>
	</div>

<div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_date" ><div style="text-align: left" >April 14, 2026</div></div><div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_excerpt" ><div style="text-align: left" ><p>Thirty years after Atlanta hosted the 1996 Centennial Olympic Games, Georgia State looks back in a two-part series at how that moment reshaped both the city and the university for generations to come.</p>
</div></div></div></div>	</div>
</div>
</div><div class="vc_clearfix"></div></div></div></div>	</div>
</div><!-- vc_grid end -->
</div></div></div></div><div class="vc_row wpb_row vc_row-fluid vc_custom_1587477528698 vc_column-gap-35 vc_row-o-equal-height vc_row-flex"><div class="shadow-box wpb_column vc_column_container vc_col-sm-9 vc_col-md-8"><div class="vc_column-inner"><div class="wpb_wrapper"><div class="vc_row wpb_row vc_inner vc_row-fluid"><div class="wpb_column vc_column_container vc_col-sm-12"><div class="vc_column-inner"><div class="wpb_wrapper"><div id="ultimate-heading-807469fe346a21289" class="uvc-heading ult-adjust-bottom-margin ultimate-heading-807469fe346a21289 uvc-6385 " data-hspacer="no_spacer"  data-halign="left" style="text-align:left"><div class="uvc-heading-spacer no_spacer" style="top"></div><div class="uvc-main-heading ult-responsive"  data-ultimate-target='.uvc-heading.ultimate-heading-807469fe346a21289 h4'  data-responsive-json-new='{"font-size":"","line-height":""}' ><h4 style="font-weight:normal;margin-bottom:15px;">Top Stories</h4></div></div></div></div></div></div><div class="vc_row wpb_row vc_inner vc_row-fluid"><div class="wpb_column vc_column_container vc_col-sm-12 vc_hidden-sm vc_hidden-xs"><div class="vc_column-inner"><div class="wpb_wrapper"><!-- vc_grid start -->
<div class="vc_grid-container-wrapper vc_clearfix" >
	<div class="vc_grid-container vc_clearfix wpb_content_element vc_basic_grid news-latest" data-initial-loading-animation="none" data-vc-grid-settings="{&quot;page_id&quot;:682,&quot;style&quot;:&quot;pagination&quot;,&quot;action&quot;:&quot;vc_get_vc_grid_data&quot;,&quot;shortcode_id&quot;:&quot;1777651571216-bd71c426-acee-3&quot;,&quot;items_per_page&quot;:&quot;5&quot;,&quot;auto_play&quot;:false,&quot;gap&quot;:0,&quot;speed&quot;:-1000,&quot;loop&quot;:&quot;&quot;,&quot;animation_in&quot;:&quot;&quot;,&quot;animation_out&quot;:&quot;&quot;,&quot;arrows_design&quot;:&quot;none&quot;,&quot;arrows_color&quot;:&quot;blue&quot;,&quot;arrows_position&quot;:&quot;inside&quot;,&quot;paging_design&quot;:&quot;pagination_square&quot;,&quot;paging_color&quot;:&quot;black&quot;,&quot;tag&quot;:&quot;vc_basic_grid&quot;}" data-vc-request="https://news.gsu.edu/wp-admin/admin-ajax.php" data-vc-post-id="682" data-vc-public-nonce="bc96b7788b">
			</div>
</div><!-- vc_grid end -->
</div></div></div></div><div class="vc_row wpb_row vc_inner vc_row-fluid"><div class="wpb_column vc_column_container vc_col-sm-12 vc_hidden-lg vc_hidden-md"><div class="vc_column-inner"><div class="wpb_wrapper"><!-- vc_grid start -->
<div class="vc_grid-container-wrapper vc_clearfix" >
	<div class="vc_grid-container vc_clearfix wpb_content_element vc_basic_grid mobile-news-latest" data-initial-loading-animation="none" data-vc-grid-settings="{&quot;page_id&quot;:682,&quot;style&quot;:&quot;load-more&quot;,&quot;action&quot;:&quot;vc_get_vc_grid_data&quot;,&quot;shortcode_id&quot;:&quot;1777651571216-d48215da-d61f-7&quot;,&quot;items_per_page&quot;:&quot;5&quot;,&quot;btn_data&quot;:{&quot;title&quot;:&quot;LOAD MORE&quot;,&quot;style&quot;:&quot;flat&quot;,&quot;gradient_color_1&quot;:&quot;flat&quot;,&quot;gradient_color_2&quot;:&quot;flat&quot;,&quot;gradient_custom_color_1&quot;:&quot;flat&quot;,&quot;gradient_custom_color_2&quot;:&quot;flat&quot;,&quot;gradient_text_color&quot;:&quot;flat&quot;,&quot;custom_background&quot;:&quot;#ededed&quot;,&quot;custom_text&quot;:&quot;#666&quot;,&quot;outline_custom_color&quot;:&quot;#666&quot;,&quot;outline_custom_hover_background&quot;:&quot;#666&quot;,&quot;outline_custom_hover_text&quot;:&quot;#fff&quot;,&quot;shape&quot;:&quot;square&quot;,&quot;color&quot;:&quot;blue&quot;,&quot;size&quot;:&quot;md&quot;,&quot;align&quot;:&quot;inline&quot;,&quot;button_block&quot;:&quot;&quot;,&quot;add_icon&quot;:&quot;&quot;,&quot;i_align&quot;:&quot;left&quot;,&quot;i_type&quot;:&quot;fontawesome&quot;,&quot;i_icon_fontawesome&quot;:&quot;fa fa-adjust&quot;,&quot;i_icon_openiconic&quot;:&quot;vc-oi vc-oi-dial&quot;,&quot;i_icon_typicons&quot;:&quot;typcn typcn-adjust-brightness&quot;,&quot;i_icon_entypo&quot;:&quot;entypo-icon entypo-icon-note&quot;,&quot;i_icon_linecons&quot;:&quot;vc_li vc_li-heart&quot;,&quot;i_icon_monosocial&quot;:&quot;vc_li vc_li-heart&quot;,&quot;i_icon_material&quot;:&quot;vc_li vc_li-heart&quot;,&quot;i_icon_pixelicons&quot;:&quot;vc_pixel_icon vc_pixel_icon-alert&quot;,&quot;el_id&quot;:&quot;&quot;,&quot;custom_onclick&quot;:&quot;&quot;,&quot;custom_onclick_code&quot;:&quot;&quot;},&quot;tag&quot;:&quot;vc_basic_grid&quot;}" data-vc-request="https://news.gsu.edu/wp-admin/admin-ajax.php" data-vc-post-id="682" data-vc-public-nonce="bc96b7788b">
		<ul class="vc_grid-filter vc_clearfix vc_grid-filter-filled vc_grid-filter-size-xs vc_grid-filter-left vc_grid-filter-color-grey" data-vc-grid-filter="category"><li class="vc_active vc_grid-filter-item"><span data-vc-grid-filter-value="*">All</span></li><li class="vc_grid-filter-item"><span data-vc-grid-filter-value=".vc_grid-term-313">Arts &amp; Culture</span><!-- fix whitespace
				--></li><li class="vc_grid-filter-item"><span data-vc-grid-filter-value=".vc_grid-term-44069">Beyond College to Career</span><!-- fix whitespace
				--></li><li class="vc_grid-filter-item"><span data-vc-grid-filter-value=".vc_grid-term-346">Education &amp; Leadership</span><!-- fix whitespace
				--></li><li class="vc_grid-filter-item"><span data-vc-grid-filter-value=".vc_grid-term-354">Science &amp; Technology</span><!-- fix whitespace
				--></li></ul>	<!-- for responsive vc_responsive !-->
	<div class="vc_grid-filter-select vc_grid-filter-left vc_grid-filter-color-grey" data-vc-grid-filter-select="category">
		<div class="vc_grid-styled-select"><select data-filter="category">
				<option class="vc_active" value="*">All&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</option>
				<option value=".vc_grid-term-313">Arts &amp; Culture&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</option><option value=".vc_grid-term-44069">Beyond College to Career&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</option><option value=".vc_grid-term-346">Education &amp; Leadership&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</option><option value=".vc_grid-term-354">Science &amp; Technology&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</option>			</select><i class="vc_arrow-icon-navicon"></i>
		</div>
	</div>
<div class="vc_grid vc_row vc_pageable-wrapper vc_hook_hover" data-vc-pageable-content="true"><div class="vc_pageable-slide-wrapper vc_clearfix" data-vc-grid-content="true"><div class="vc_grid-item vc_clearfix vc_col-sm-12 vc_grid-item-zone-c-bottom vc_grid-term-44069 vc_grid-term-346"><div class="vc_grid-item-mini vc_clearfix "><div class="vc_gitem-animated-block "></div>
<div class="vc_gitem-zone vc_gitem-zone-c">
	<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-">
	<div class="wpb_single_image wpb_content_element vc_align_">
		<figure class="wpb_wrapper vc_figure">
			<a href="https://news.gsu.edu/2026/05/08/alexis-alvarez-b-s-ed-26-creating-inclusive-engaging-learning-experiences/" class="vc_gitem-link vc_single_image-wrapper vc_box_border_grey" title="Future Educator Learns to Create Inclusive, Engaging Learning Experiences"><img width="800" height="533" src="https://news.gsu.edu/files/2026/05/Alexis-Alvarez-2026-graduate-5.26.jpg" class="vc_single_image-img attachment-medium" alt="Alexis Alvarez poses for a photo in her graduation regalia, holding her cap that she decorated to look like a classroom with Peanuts characters in it" srcset="https://news.gsu.edu/files/2026/05/Alexis-Alvarez-2026-graduate-5.26.jpg 800w, https://news.gsu.edu/files/2026/05/Alexis-Alvarez-2026-graduate-5.26-300x200.jpg 300w, https://news.gsu.edu/files/2026/05/Alexis-Alvarez-2026-graduate-5.26-768x512.jpg 768w, https://news.gsu.edu/files/2026/05/Alexis-Alvarez-2026-graduate-5.26-450x300.jpg 450w" sizes="(max-width: 800px) 100vw, 800px" /></a>
		</figure>
	</div>

<div class="vc_gitem-post-data vc_gitem-post-data-source-post_categories vc_grid-filter vc_clearfix vc_grid-filter-comma vc_grid-filter-size-md vc_grid-filter-center vc_grid-filter-color-grey"><div class="vc_grid-filter-item vc_gitem-post-category-name"><span class="vc_gitem-post-category-name"><a href="https://news.gsu.edu/category/beyond-college-to-career/" alt="View all posts in Beyond College to Career" class="vc_gitem-link">Beyond College to Career</a></span></div><div class="vc_grid-filter-item vc_gitem-post-category-name"><span class="vc_gitem-post-category-name"><a href="https://news.gsu.edu/category/education-leadership/" alt="View all posts in Education &amp; Leadership" class="vc_gitem-link">Education &amp; Leadership</a></span></div></div>
<div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_date" ><div style="text-align: left" >May 8, 2026</div></div></div></div><div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-"><div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_title" ><h4 style="text-align: left" ><a href="https://news.gsu.edu/2026/05/08/alexis-alvarez-b-s-ed-26-creating-inclusive-engaging-learning-experiences/" class="vc_gitem-link" title="Future Educator Learns to Create Inclusive, Engaging Learning Experiences">Future Educator Learns to Create Inclusive, Engaging Learning Experiences</a></h4></div><div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_excerpt" ><div style="text-align: left" ><p>Alexis Alvarez (B.S.Ed. '26) was drawn to Georgia State University for the field experiences offered in the elementary education bachelor's program.</p>
</div></div><div class="vc_separator wpb_content_element vc_separator_align_center vc_sep_width_90 vc_sep_pos_align_center vc_separator_no_text vc_sep_color_grey" ><span class="vc_sep_holder vc_sep_holder_l"><span  class="vc_sep_line"></span></span><span class="vc_sep_holder vc_sep_holder_r"><span  class="vc_sep_line"></span></span>
</div></div></div>	</div>
</div>
</div><div class="vc_clearfix"></div></div><div class="vc_grid-item vc_clearfix vc_col-sm-12 vc_grid-item-zone-c-bottom vc_grid-term-354"><div class="vc_grid-item-mini vc_clearfix "><div class="vc_gitem-animated-block "></div>
<div class="vc_gitem-zone vc_gitem-zone-c">
	<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-">
	<div class="wpb_single_image wpb_content_element vc_align_">
		<figure class="wpb_wrapper vc_figure">
			<a href="https://news.gsu.edu/2026/05/06/perimeter-college-graduate-driven-by-desire-to-serve-others/" class="vc_gitem-link vc_single_image-wrapper vc_box_border_grey" title="Perimeter College Graduate Driven to Engineer Housing Solutions"><img width="960" height="720" src="https://news.gsu.edu/files/2014/09/NDEYE-SARR_800x600--960x720.jpg" class="vc_single_image-img attachment-medium" alt="Student sits outside on college quad" srcset="https://news.gsu.edu/files/2014/09/NDEYE-SARR_800x600--960x720.jpg 960w, https://news.gsu.edu/files/2014/09/NDEYE-SARR_800x600--768x576.jpg 768w, https://news.gsu.edu/files/2014/09/NDEYE-SARR_800x600--1500x1125.jpg 1500w, https://news.gsu.edu/files/2014/09/NDEYE-SARR_800x600--400x300.jpg 400w, https://news.gsu.edu/files/2014/09/NDEYE-SARR_800x600--800x600.jpg 800w, https://news.gsu.edu/files/2014/09/NDEYE-SARR_800x600--1200x900.jpg 1200w, https://news.gsu.edu/files/2014/09/NDEYE-SARR_800x600-.jpg 1600w" sizes="(max-width: 960px) 100vw, 960px" /></a>
		</figure>
	</div>

<div class="vc_gitem-post-data vc_gitem-post-data-source-post_categories vc_grid-filter vc_clearfix vc_grid-filter-comma vc_grid-filter-size-md vc_grid-filter-center vc_grid-filter-color-grey"><div class="vc_grid-filter-item vc_gitem-post-category-name"><span class="vc_gitem-post-category-name"><a href="https://news.gsu.edu/category/science-technology/" alt="View all posts in Science &amp; Technology" class="vc_gitem-link">Science &amp; Technology</a></span></div></div>
<div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_date" ><div style="text-align: left" >May 6, 2026</div></div></div></div><div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-"><div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_title" ><h4 style="text-align: left" ><a href="https://news.gsu.edu/2026/05/06/perimeter-college-graduate-driven-by-desire-to-serve-others/" class="vc_gitem-link" title="Perimeter College Graduate Driven to Engineer Housing Solutions">Perimeter College Graduate Driven to Engineer Housing Solutions</a></h4></div><div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_excerpt" ><div style="text-align: left" ><p>Graduating this spring from Georgia State University's Perimeter College with an associate degree in engineering, Ndeye Sarr (A.S. '26) is an Honors College student, a semifinalist for the Jack Kent Cooke Undergraduate Transfer Scholarship and a recent recipient of Perimeter’s STAR Award in engineering.</p>
</div></div><div class="vc_separator wpb_content_element vc_separator_align_center vc_sep_width_90 vc_sep_pos_align_center vc_separator_no_text vc_sep_color_grey" ><span class="vc_sep_holder vc_sep_holder_l"><span  class="vc_sep_line"></span></span><span class="vc_sep_holder vc_sep_holder_r"><span  class="vc_sep_line"></span></span>
</div></div></div>	</div>
</div>
</div><div class="vc_clearfix"></div></div><div class="vc_grid-item vc_clearfix vc_col-sm-12 vc_grid-item-zone-c-bottom vc_grid-term-313"><div class="vc_grid-item-mini vc_clearfix "><div class="vc_gitem-animated-block "></div>
<div class="vc_gitem-zone vc_gitem-zone-c">
	<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-">
	<div class="wpb_single_image wpb_content_element vc_align_">
		<figure class="wpb_wrapper vc_figure">
			<a href="https://news.gsu.edu/2026/05/06/journalism-major-finds-inspiration-opportunity-in-atlanta/" class="vc_gitem-link vc_single_image-wrapper vc_box_border_grey" title="Journalism Major Finds Inspiration, Opportunity in Atlanta"><img width="768" height="512" src="https://news.gsu.edu/files/2014/09/Philena-Styles_768.jpg" class="vc_single_image-img attachment-medium" alt="Portrait photo of Philena Styles sitting on steps with a mortarboard in hand." srcset="https://news.gsu.edu/files/2014/09/Philena-Styles_768.jpg 768w, https://news.gsu.edu/files/2014/09/Philena-Styles_768-300x200.jpg 300w, https://news.gsu.edu/files/2014/09/Philena-Styles_768-450x300.jpg 450w" sizes="(max-width: 768px) 100vw, 768px" /></a>
		</figure>
	</div>

<div class="vc_gitem-post-data vc_gitem-post-data-source-post_categories vc_grid-filter vc_clearfix vc_grid-filter-comma vc_grid-filter-size-md vc_grid-filter-center vc_grid-filter-color-grey"><div class="vc_grid-filter-item vc_gitem-post-category-name"><span class="vc_gitem-post-category-name"><a href="https://news.gsu.edu/category/arts-culture/" alt="View all posts in Arts &amp; Culture" class="vc_gitem-link">Arts &amp; Culture</a></span></div></div>
<div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_date" ><div style="text-align: left" >May 6, 2026</div></div></div></div><div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-"><div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_title" ><h4 style="text-align: left" ><a href="https://news.gsu.edu/2026/05/06/journalism-major-finds-inspiration-opportunity-in-atlanta/" class="vc_gitem-link" title="Journalism Major Finds Inspiration, Opportunity in Atlanta">Journalism Major Finds Inspiration, Opportunity in Atlanta</a></h4></div><div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_excerpt" ><div style="text-align: left" ><p>A native of the Bahamas, journalism major Philena Styles (B.A. '26) found her footing in faith, community and academics at Georgia State. </p>
</div></div><div class="vc_separator wpb_content_element vc_separator_align_center vc_sep_width_90 vc_sep_pos_align_center vc_separator_no_text vc_sep_color_grey" ><span class="vc_sep_holder vc_sep_holder_l"><span  class="vc_sep_line"></span></span><span class="vc_sep_holder vc_sep_holder_r"><span  class="vc_sep_line"></span></span>
</div></div></div>	</div>
</div>
</div><div class="vc_clearfix"></div></div><div class="vc_grid-item vc_clearfix vc_col-sm-12 vc_grid-item-zone-c-bottom vc_grid-term-44069 vc_grid-term-346"><div class="vc_grid-item-mini vc_clearfix "><div class="vc_gitem-animated-block "></div>
<div class="vc_gitem-zone vc_gitem-zone-c">
	<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-">
	<div class="wpb_single_image wpb_content_element vc_align_">
		<figure class="wpb_wrapper vc_figure">
			<a href="https://news.gsu.edu/2026/05/06/state-farm-scholar-natalia-donahue-turns-passion-for-teaching-into-career-path/" class="vc_gitem-link vc_single_image-wrapper vc_box_border_grey" title="State Farm Scholar Turns Passion for Teaching Into Career Path"><img width="800" height="533" src="https://news.gsu.edu/files/2026/05/Natalia-Donahue-2026-graduate-5.26.jpg" class="vc_single_image-img attachment-medium" alt="Natalia Donahue reads the children&#039;s book &quot;Dragons Love Tacos&quot; in front of the Hurt Park fountain with other children&#039;s books on either side of her" srcset="https://news.gsu.edu/files/2026/05/Natalia-Donahue-2026-graduate-5.26.jpg 800w, https://news.gsu.edu/files/2026/05/Natalia-Donahue-2026-graduate-5.26-300x200.jpg 300w, https://news.gsu.edu/files/2026/05/Natalia-Donahue-2026-graduate-5.26-768x512.jpg 768w, https://news.gsu.edu/files/2026/05/Natalia-Donahue-2026-graduate-5.26-450x300.jpg 450w" sizes="(max-width: 800px) 100vw, 800px" /></a>
		</figure>
	</div>

<div class="vc_gitem-post-data vc_gitem-post-data-source-post_categories vc_grid-filter vc_clearfix vc_grid-filter-comma vc_grid-filter-size-md vc_grid-filter-center vc_grid-filter-color-grey"><div class="vc_grid-filter-item vc_gitem-post-category-name"><span class="vc_gitem-post-category-name"><a href="https://news.gsu.edu/category/beyond-college-to-career/" alt="View all posts in Beyond College to Career" class="vc_gitem-link">Beyond College to Career</a></span></div><div class="vc_grid-filter-item vc_gitem-post-category-name"><span class="vc_gitem-post-category-name"><a href="https://news.gsu.edu/category/education-leadership/" alt="View all posts in Education &amp; Leadership" class="vc_gitem-link">Education &amp; Leadership</a></span></div></div>
<div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_date" ><div style="text-align: left" >May 6, 2026</div></div></div></div><div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-"><div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_title" ><h4 style="text-align: left" ><a href="https://news.gsu.edu/2026/05/06/state-farm-scholar-natalia-donahue-turns-passion-for-teaching-into-career-path/" class="vc_gitem-link" title="State Farm Scholar Turns Passion for Teaching Into Career Path">State Farm Scholar Turns Passion for Teaching Into Career Path</a></h4></div><div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_excerpt" ><div style="text-align: left" ><p>Thanks to the State Farm Scholars program and the State Farm Transition Scholarship, Natalia Donahue (B.S.Ed. '26) is prepared to become an elementary school teacher.</p>
</div></div><div class="vc_separator wpb_content_element vc_separator_align_center vc_sep_width_90 vc_sep_pos_align_center vc_separator_no_text vc_sep_color_grey" ><span class="vc_sep_holder vc_sep_holder_l"><span  class="vc_sep_line"></span></span><span class="vc_sep_holder vc_sep_holder_r"><span  class="vc_sep_line"></span></span>
</div></div></div>	</div>
</div>
</div><div class="vc_clearfix"></div></div><div class="vc_grid-item vc_clearfix vc_col-sm-12 vc_grid-item-zone-c-bottom vc_grid-term-44069"><div class="vc_grid-item-mini vc_clearfix "><div class="vc_gitem-animated-block "></div>
<div class="vc_gitem-zone vc_gitem-zone-c">
	<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-">
	<div class="wpb_single_image wpb_content_element vc_align_">
		<figure class="wpb_wrapper vc_figure">
			<a href="https://news.gsu.edu/2026/05/06/how-an-mph-opened-doors-beyond-a-traditional-public-health-career-for-hayley-hamilton/" class="vc_gitem-link vc_single_image-wrapper vc_box_border_grey" title="How an MPH Opened Doors Beyond a Traditional Public Health Career for Hayley Hamilton"><img width="800" height="600" src="https://news.gsu.edu/files/2014/09/Hayley-Hamilton.png" class="vc_single_image-img attachment-medium" alt="Hayley Hamilton portrait" srcset="https://news.gsu.edu/files/2014/09/Hayley-Hamilton.png 800w, https://news.gsu.edu/files/2014/09/Hayley-Hamilton-768x576.png 768w, https://news.gsu.edu/files/2014/09/Hayley-Hamilton-400x300.png 400w" sizes="(max-width: 800px) 100vw, 800px" /></a>
		</figure>
	</div>

<div class="vc_gitem-post-data vc_gitem-post-data-source-post_categories vc_grid-filter vc_clearfix vc_grid-filter-comma vc_grid-filter-size-md vc_grid-filter-center vc_grid-filter-color-grey"><div class="vc_grid-filter-item vc_gitem-post-category-name"><span class="vc_gitem-post-category-name"><a href="https://news.gsu.edu/category/beyond-college-to-career/" alt="View all posts in Beyond College to Career" class="vc_gitem-link">Beyond College to Career</a></span></div></div>
<div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_date" ><div style="text-align: left" >May 6, 2026</div></div></div></div><div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-"><div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_title" ><h4 style="text-align: left" ><a href="https://news.gsu.edu/2026/05/06/how-an-mph-opened-doors-beyond-a-traditional-public-health-career-for-hayley-hamilton/" class="vc_gitem-link" title="How an MPH Opened Doors Beyond a Traditional Public Health Career for Hayley Hamilton">How an MPH Opened Doors Beyond a Traditional Public Health Career for Hayley Hamilton</a></h4></div><div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_excerpt" ><div style="text-align: left" ><p>Hayley Hamilton (MPH '17) found her path in public health while searching for a career that aligned with both her interests and values. </p>
</div></div><div class="vc_separator wpb_content_element vc_separator_align_center vc_sep_width_90 vc_sep_pos_align_center vc_separator_no_text vc_sep_color_grey" ><span class="vc_sep_holder vc_sep_holder_l"><span  class="vc_sep_line"></span></span><span class="vc_sep_holder vc_sep_holder_r"><span  class="vc_sep_line"></span></span>
</div></div></div>	</div>
</div>
</div><div class="vc_clearfix"></div></div></div><div class="vc_pageable-load-more-btn" data-vc-grid-load-more-btn="true">	<div class="vc_btn3-container  vc_grid-btn-load_more vc_btn3-inline" >
		<a href="javascript:;" class="vc_general vc_btn3 vc_btn3-size-md vc_btn3-shape-square vc_btn3-style-flat vc_btn3-color-blue"  title="LOAD MORE">LOAD MORE</a></div>


</div></div>	</div>
</div><!-- vc_grid end -->
</div></div></div></div></div></div></div><div class="shadow-box wpb_column vc_column_container vc_col-sm-3 vc_col-md-4"><div class="vc_column-inner"><div class="wpb_wrapper"><div class="vc_row wpb_row vc_inner vc_row-fluid"><div class="wpb_column vc_column_container vc_col-sm-12"><div class="vc_column-inner"><div class="wpb_wrapper"><div id="ultimate-heading-346569fe346a55c3a" class="uvc-heading ult-adjust-bottom-margin ultimate-heading-346569fe346a55c3a uvc-8532 " data-hspacer="no_spacer"  data-halign="left" style="text-align:left"><div class="uvc-heading-spacer no_spacer" style="top"></div><div class="uvc-main-heading ult-responsive"  data-ultimate-target='.uvc-heading.ultimate-heading-346569fe346a55c3a h4'  data-responsive-json-new='{"font-size":"","line-height":""}' ><h4 style="font-weight:normal;">Search Georgia State News</h4></div></div><div class="vc_empty_space"   style="height: 16px" ><span class="vc_empty_space_inner"></span></div>
<div class='asp_w asp_m asp_m_31 asp_m_31_1 wpdreams_asp_sc wpdreams_asp_sc-31 ajaxsearchpro asp_main_container  asp_non_compact'
     data-id="31"
          data-instance="1"
     id='ajaxsearchpro31_1'>
<div class="probox">
    
    <div class='promagnifier'>
        	    <div class='asp_text_button hiddend'>
		    Search	    </div>
        <div class='innericon'>
            <svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><path d="M448.225 394.243l-85.387-85.385c16.55-26.08 26.146-56.986 26.146-90.094 0-92.99-75.652-168.64-168.643-168.64-92.988 0-168.64 75.65-168.64 168.64s75.65 168.64 168.64 168.64c31.466 0 60.94-8.67 86.176-23.734l86.14 86.142c36.755 36.754 92.355-18.783 55.57-55.57zm-344.233-175.48c0-64.155 52.192-116.35 116.35-116.35s116.353 52.194 116.353 116.35S284.5 335.117 220.342 335.117s-116.35-52.196-116.35-116.352zm34.463-30.26c34.057-78.9 148.668-69.75 170.248 12.863-43.482-51.037-119.984-56.532-170.248-12.862z"/></svg>        </div>
	    <div class="asp_clear"></div>
    </div>

    
    
    <div class='prosettings'  data-opened=0>
                <div class='innericon'>
            <svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><path id="menu-3-icon" d="M275 163.5H50v-65h225v65zm0 60H50v65h225v-65zm0 125H50v65h225v-65zm44.105-130.592l71.447 87.12L462 217.91H319.105z"/></svg>        </div>
    </div>

    
    
    <div class='proinput'>
        <form action='#' autocomplete="off" aria-label="Search form 31">
            <input type='search' class='orig'
                   placeholder='Search here...'
                   name='phrase' value=''
                   aria-label="Search input 31"
                   autocomplete="off"/>
            <input type='text' class='autocomplete' name='phrase' value=''
                   aria-label="Search autocomplete, ignore please"
                   aria-hidden="true"
                   autocomplete="off" disabled/>
            <input type='submit'
                   aria-hidden="true"
                   aria-label="Hidden button"
                   style='width:0; height: 0; visibility: hidden;'>
        </form>
    </div>

    
    
    <div class='proloading'>
                        <div class="asp_loader">
            <div class="asp_loader-inner asp_ball-grid-pulse">
            
                <div></div>
                
                <div></div>
                
                <div></div>
                
                <div></div>
                
                <div></div>
                
                <div></div>
                
                <div></div>
                
                <div></div>
                
                <div></div>
                            </div>
        </div>
                    </div>

            <div class='proclose'>
            <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px"
                 y="0px"
                 width="512px" height="512px" viewBox="0 0 512 512" enable-background="new 0 0 512 512"
                 xml:space="preserve">
            <polygon id="x-mark-icon"
                     points="438.393,374.595 319.757,255.977 438.378,137.348 374.595,73.607 255.995,192.225 137.375,73.622 73.607,137.352 192.246,255.983 73.622,374.625 137.352,438.393 256.002,319.734 374.652,438.378 "/>
            </svg>
        </div>
    
    
</div><div id='ajaxsearchprores31_1' class='asp_w asp_r asp_r_31 asp_r_31_1 vertical ajaxsearchpro wpdreams_asp_sc wpdreams_asp_sc-31'
     data-id="31"
     data-instance="1">

    
    
    
    <div class="results">

        
        <div class="resdrg">
        </div>

        
    </div>

    
                    <div class="asp_showmore_container">
            <p class='showmore'>
                <a class='asp_showmore'>More results... <span></span></a>
            </p>
            <div class="asp_moreres_loader" style="display: none;">
                <div class="asp_moreres_loader-inner"></div>
            </div>
        </div>
            
    

    <div class="asp_res_loader hiddend">
                    <div class="asp_loader">
                <div class="asp_loader-inner asp_ball-grid-pulse">
                
                    <div></div>
                    
                    <div></div>
                    
                    <div></div>
                    
                    <div></div>
                    
                    <div></div>
                    
                    <div></div>
                    
                    <div></div>
                    
                    <div></div>
                    
                    <div></div>
                                    </div>
            </div>
            </div>
</div>    <div id='ajaxsearchprosettings31_1' class="asp_w asp_s asp_s_31 asp_s_31_1 wpdreams_asp_sc wpdreams_asp_sc-31 ajaxsearchpro searchsettings"
    data-id="31"
    data-instance="1">
<form name='options' class="asp-fss-column" autocomplete = 'off'>
        <input type="hidden" style="display:none;" name="current_page_id" value="682">
            <fieldset class="">
            <legend>Generic filters</legend>
    
    <div class="asp_option_inner hiddend">
        <input type='hidden' name='qtranslate_lang'
               value='0'/>
    </div>

    
	

                    <div class="asp_option">
            <div class="asp_option_inner">
                <input type="checkbox" value="exact" id="set_exact31_1"
                       aria-label="Exact matches only"
                       name="asp_gen[]" />
                <label aria-hidden="true"
                       for="set_exact31_1">
                    Hidden label                </label>
            </div>
            <div class="asp_option_label">
                Exact matches only            </div>
        </div>
                                <div class="asp_option hiddend" aria-hidden="true">
            <div class="asp_option_inner">
                <input type="checkbox" value="title" id="set_title31_1"
                       name="asp_gen[]"  checked="checked"/>
                <label for="set_title31_1">
                    Hidden label                </label>
            </div>
        </div>
                    <div class="asp_option hiddend" aria-hidden="true">
            <div class="asp_option_inner">
                <input type="checkbox" value="content" id="set_content31_1"
                       name="asp_gen[]"  checked="checked"/>
                <label for="set_content31_1">
                    Hidden label                </label>
            </div>
        </div>
                    <div class="asp_option hiddend" aria-hidden="true">
            <div class="asp_option_inner">
                <input type="checkbox" value="excerpt" id="set_excerpt31_1"
                       name="asp_gen[]"  checked="checked"/>
                <label for="set_excerpt31_1">
                    Hidden label                </label>
            </div>
        </div>
    </fieldset><fieldset class="asp_sett_scroll hiddend asp_checkboxes_filter_box">
        <legend>Filter by Custom Post Type</legend>
            <div class="asp_option_inner hiddend" aria-hidden="true">
        <input type="checkbox" value="post"
               id="31_1customset_31_11"
               name="customset[]" checked="checked"/>
        <label for="31_1customset_31_11">
            Hidden label        </label>
    </div>
    <div class="asp_option_label hiddend"></div>
        <div class="asp_option_inner hiddend" aria-hidden="true">
        <input type="checkbox" value="page"
               id="31_1customset_31_12"
               name="customset[]" checked="checked"/>
        <label for="31_1customset_31_12">
            Hidden label        </label>
    </div>
    <div class="asp_option_label hiddend"></div>
    </fieldset>
<fieldset>
            <div>
                <legend>Content from</legend>
                <textarea class="asp_datepicker_format"
                  aria-hidden="true"
                  aria-label="Content from"
                  style="display:none !important;">mm/dd/yy</textarea>
        <input type="text"
               aria-label="Content from"
               placeholder="Choose date"
               class="asp_datepicker" name="post_date_from_real" value="-4y -0m -0d">
        <input type="hidden" class="asp_datepicker_hidden" name="post_date_from" value="">
        </div>
    
            <div>
                <legend style="margin-top: 10px;">Content to</legend>
                <textarea class="asp_datepicker_format"
                  aria-hidden="true"
                  aria-label="Content to"
                  style="display:none !important;">mm/dd/yy</textarea>
        <input type="text"
               aria-label="Content to"
               placeholder="Choose date"
               class="asp_datepicker" name="post_date_to_real" value="-0y -0m -0d">
        <input type="hidden" class="asp_datepicker_hidden" name="post_date_to" value="">
        </div>
    </fieldset>    <div style="clear:both;"></div>
</form>
</div>

</div>
<div class='asp_hidden_data' id="asp_hidden_data_31_1" style="display:none;">

    <div class='asp_item_overlay'>
        <div class='asp_item_inner'>
            <svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><path d="M448.225 394.243l-85.387-85.385c16.55-26.08 26.146-56.986 26.146-90.094 0-92.99-75.652-168.64-168.643-168.64-92.988 0-168.64 75.65-168.64 168.64s75.65 168.64 168.64 168.64c31.466 0 60.94-8.67 86.176-23.734l86.14 86.142c36.755 36.754 92.355-18.783 55.57-55.57zm-344.233-175.48c0-64.155 52.192-116.35 116.35-116.35s116.353 52.194 116.353 116.35S284.5 335.117 220.342 335.117s-116.35-52.196-116.35-116.352zm34.463-30.26c34.057-78.9 148.668-69.75 170.248 12.863-43.482-51.037-119.984-56.532-170.248-12.862z"/></svg>                    </div>
    </div>

</div>        <style type="text/css">
        /* User defined Ajax Search Pro Custom CSS */
        .asp-ui .ui-widget-header {
    background: #373e57 !important;
    color: #e5e5e5;
}
.asp-ui.ui-datepicker .ui-state-active {
    background: #373e57 !important;
}    </style>
    <div class="asp_init_data" style="display:none !important;" id="asp_init_id_31_1" data-aspdata="ew0KICAgICJob21ldXJsIjogImh0dHBzOi8vbmV3cy5nc3UuZWR1LyIsDQogICAgInJlc3VsdHN0eXBlIjogInZlcnRpY2FsIiwNCiAgICAicmVzdWx0c3Bvc2l0aW9uIjogImhvdmVyIiwNCiAgICAicmVzdWx0c1NuYXBUbyI6ICJsZWZ0IiwNCiAgICAicmVzdWx0cyI6IHsNCiAgICAgICAgIndpZHRoIjogImF1dG8iLA0KICAgICAgICAid2lkdGhfdGFibGV0IjogImF1dG8iLA0KICAgICAgICAid2lkdGhfcGhvbmUiOiAiYXV0byINCiAgICB9LA0KICAgICJpdGVtc2NvdW50IjogNCwNCiAgICAiaW1hZ2V3aWR0aCI6IDcwLA0KICAgICJpbWFnZWhlaWdodCI6IDcwLA0KICAgICJyZXN1bHRpdGVtaGVpZ2h0IjogImF1dG8iLA0KICAgICJzaG93YXV0aG9yIjogMCwNCiAgICAic2hvd2RhdGUiOiAwLA0KICAgICJzaG93ZGVzY3JpcHRpb24iOiAxLA0KICAgICJjaGFyY291bnQiOiAgMCwNCiAgICAiZGVmYXVsdEltYWdlIjogImh0dHBzOi8vbmV3cy5nc3UuZWR1L3dwLWNvbnRlbnQvcGx1Z2lucy9hamF4LXNlYXJjaC1wcm8vaW1nL2RlZmF1bHQuanBnIiwNCiAgICAiaGlnaGxpZ2h0IjogMSwNCiAgICAiaGlnaGxpZ2h0d2hvbGV3b3JkcyI6IDEsDQogICAgIm9wZW5Ub0JsYW5rIjogMCwNCiAgICAic2Nyb2xsVG9SZXN1bHRzIjogMCwNCiAgICAicmVzdWx0YXJlYWNsaWNrYWJsZSI6IDEsDQogICAgImF1dG9jb21wbGV0ZSI6IHsNCiAgICAgICAgImVuYWJsZWQiOiAxLA0KICAgICAgICAiZ29vZ2xlT25seSI6IDEsDQogICAgICAgICJsYW5nIjogImVuIiwNCiAgICAgICAgIm1vYmlsZSI6IDEgICAgfSwNCiAgICAidHJpZ2dlcm9udHlwZSI6IDEsDQogICAgInRyaWdnZXJfb25fY2xpY2siOiAxLA0KICAgICJ0cmlnZ2VyT25GYWNldENoYW5nZSI6IDEsDQogICAgInRyaWdnZXIiOiB7DQogICAgICAgICJkZWxheSI6IDMwMCwNCiAgICAgICAgImF1dG9jb21wbGV0ZV9kZWxheSI6IDMxMCAgICB9LA0KICAgICJvdmVycmlkZXdwZGVmYXVsdCI6IDEsDQogICAgIm92ZXJyaWRlX21ldGhvZCI6ICJnZXQiLA0KICAgICJyZWRpcmVjdG9uY2xpY2siOiAwLA0KICAgICJyZWRpcmVjdENsaWNrVG8iOiAiYWpheF9zZWFyY2giLA0KICAgICJyZWRpcmVjdENsaWNrTG9jIjogInNhbWUiLA0KICAgICJyZWRpcmVjdF9vbl9lbnRlciI6IDAsDQogICAgInJlZGlyZWN0RW50ZXJUbyI6ICJhamF4X3NlYXJjaCIsDQogICAgInJlZGlyZWN0RW50ZXJMb2MiOiAic2FtZSIsDQogICAgInJlZGlyZWN0X3VybCI6ICI/cz17cGhyYXNlfSIsDQogICAgInNldHRpbmdzaW1hZ2Vwb3MiOiAicmlnaHQiLA0KICAgICJzZXR0aW5nc1Zpc2libGUiOiAwLA0KICAgICJzZXR0aW5nc0hpZGVPblJlcyI6IDAsDQogICAgImhyZXN1bHRoaWRlZGVzYyI6ICIwIiwNCiAgICAicHJlc2NvbnRhaW5lcmhlaWdodCI6ICI0MDBweCIsDQogICAgInBzaG93c3VidGl0bGUiOiAiMCIsDQogICAgInBzaG93ZGVzYyI6ICIxIiwNCiAgICAiY2xvc2VPbkRvY0NsaWNrIjogMSwNCiAgICAiaXNvdG9waWMiOiB7DQogICAgICAgICJpdGVtV2lkdGgiOiAiMjAwcHgiLA0KICAgICAgICAiaXRlbVdpZHRoVGFibGV0IjogIjIwMHB4IiwNCiAgICAgICAgIml0ZW1XaWR0aFBob25lIjogIjIwMHB4IiwNCiAgICAgICAgIml0ZW1IZWlnaHQiOiAiMjAwcHgiLA0KICAgICAgICAiaXRlbUhlaWdodFRhYmxldCI6ICIyMDBweCIsDQogICAgICAgICJpdGVtSGVpZ2h0UGhvbmUiOiAiMjAwcHgiLA0KICAgICAgICAiaWZOb0ltYWdlIjogImRlc2NyaXB0aW9uIiwNCiAgICAgICAgInBhZ2luYXRpb24iOiAxLA0KICAgICAgICAicm93cyI6IDIsDQogICAgICAgICJndXR0ZXIiOiA1LA0KICAgICAgICAic2hvd092ZXJsYXkiOiAxLA0KICAgICAgICAiYmx1ck92ZXJsYXkiOiAxLA0KICAgICAgICAiaGlkZUNvbnRlbnQiOiAxICAgIH0sDQogICAgImxvYWRlckxvY2F0aW9uIjogImF1dG8iLA0KICAgICJhbmFseXRpY3MiOiAwLA0KICAgICJhbmFseXRpY3NTdHJpbmciOiAiP2FqYXhfc2VhcmNoPXthc3BfdGVybX0iLA0KICAgICJzaG93X21vcmUiOiB7DQogICAgICAgICJlbmFibGVkIjogMSwNCiAgICAgICAgInVybCI6ICI/cz17cGhyYXNlfSIsDQogICAgICAgICJhY3Rpb24iOiAiYWpheCIsDQogICAgICAgICJsb2NhdGlvbiI6ICJzYW1lIiwNCiAgICAgICAgImluZmluaXRlIjogMSAgICB9LA0KICAgICJtb2JpbGUiOiB7DQogICAgICAgICJ0cmlnZ2VyX29uX3R5cGUiOiAxLA0KICAgICAgICAiY2xpY2tfYWN0aW9uIjogImFqYXhfc2VhcmNoIiwNCiAgICAgICAgInJldHVybl9hY3Rpb24iOiAiYWpheF9zZWFyY2giLA0KICAgICAgICAiY2xpY2tfYWN0aW9uX2xvY2F0aW9uIjogInNhbWUiLA0KICAgICAgICAicmV0dXJuX2FjdGlvbl9sb2NhdGlvbiI6ICJzYW1lIiwNCiAgICAgICAgInJlZGlyZWN0X3VybCI6ICI/cz17cGhyYXNlfSIsDQogICAgICAgICJoaWRlX2tleWJvYXJkIjogMCwNCiAgICAgICAgImZvcmNlX3Jlc19ob3ZlciI6IDAsDQogICAgICAgICJmb3JjZV9zZXR0X2hvdmVyIjogMCwNCiAgICAgICAgImZvcmNlX3NldHRfc3RhdGUiOiAiY2xvc2VkIg0KICAgIH0sDQogICAgImNvbXBhY3QiOiB7DQogICAgICAgICJlbmFibGVkIjogMCwNCiAgICAgICAgIndpZHRoIjogIjEwMCUiLA0KICAgICAgICAid2lkdGhfdGFibGV0IjogIjQ4MHB4IiwNCiAgICAgICAgIndpZHRoX3Bob25lIjogIjMyMHB4IiwNCiAgICAgICAgImNsb3NlT25NYWduaWZpZXIiOiAxLA0KICAgICAgICAiY2xvc2VPbkRvY3VtZW50IjogMCwNCiAgICAgICAgInBvc2l0aW9uIjogInN0YXRpYyIsDQogICAgICAgICJvdmVybGF5IjogMCAgICB9LA0KICAgICJzYiI6IHsNCiAgICAgICAgInJlZGlyZWN0X2FjdGlvbiI6ICJhamF4X3NlYXJjaCIsDQogICAgICAgICJyZWRpcmVjdF9sb2NhdGlvbiI6ICJzYW1lIiwNCiAgICAgICAgInJlZGlyZWN0X3VybCI6ICI/cz17cGhyYXNlfSINCiAgICB9LA0KICAgICJyYiI6IHsNCiAgICAgICAgImFjdGlvbiI6ICJub3RoaW5nIg0KICAgIH0sDQogICAgImFuaW1hdGlvbnMiOiB7DQogICAgICAgICJwYyI6IHsNCiAgICAgICAgICAgICJzZXR0aW5ncyI6IHsNCiAgICAgICAgICAgICAgICAiYW5pbSIgOiAiZmFkZWRyb3AiLA0KICAgICAgICAgICAgICAgICJkdXIiICA6IDMwMCAgICAgICAgICAgIH0sDQogICAgICAgICAgICAicmVzdWx0cyIgOiB7DQogICAgICAgICAgICAgICAgImFuaW0iIDogImZhZGVkcm9wIiwNCiAgICAgICAgICAgICAgICAiZHVyIiAgOiAzMDAgICAgICAgICAgICB9LA0KICAgICAgICAgICAgIml0ZW1zIiA6ICJmYWRlSW5Eb3duIg0KICAgICAgICB9LA0KICAgICAgICAibW9iIjogew0KICAgICAgICAgICAgInNldHRpbmdzIjogew0KICAgICAgICAgICAgICAgICJhbmltIiA6ICJmYWRlZHJvcCIsDQogICAgICAgICAgICAgICAgImR1ciIgIDogMzAwICAgICAgICAgICAgfSwNCiAgICAgICAgICAgICJyZXN1bHRzIiA6IHsNCiAgICAgICAgICAgICAgICAiYW5pbSIgOiAiZmFkZWRyb3AiLA0KICAgICAgICAgICAgICAgICJkdXIiICA6IDMwMCAgICAgICAgICAgIH0sDQogICAgICAgICAgICAiaXRlbXMiIDogInZvaWRhbmltIg0KICAgICAgICB9DQogICAgfSwNCiAgICAiY2hvc2VuIjogew0KICAgICAgICAibm9yZXMiOiAiTm8gcmVzdWx0cyBtYXRjaCINCiAgICB9LA0KICAgICJkZXRlY3RWaXNpYmlsaXR5IiA6IDAsDQogICAgImF1dG9wIjogew0KICAgICAgICAic3RhdGUiOiAiZGlzYWJsZWQiLA0KICAgICAgICAicGhyYXNlIjogIiIsDQogICAgICAgICJjb3VudCI6IDEwICAgIH0sDQogICAgInJlc1BhZ2UiOiB7DQogICAgICAgICJ1c2VBamF4IjogMCwNCiAgICAgICAgInNlbGVjdG9yIjogIiNtYWluIg0KICAgIH0sDQogICAgImZzc19sYXlvdXQiOiAiY29sdW1uIiwNCiAgICAic3RhdGlzdGljcyI6IDB9DQo="></div>
<div class="vc_empty_space"   style="height: 32px" ><span class="vc_empty_space_inner"></span></div>
</div></div></div></div><div id="ultimate-heading-465169fe346a57b39" class="uvc-heading ult-adjust-bottom-margin ultimate-heading-465169fe346a57b39 uvc-6653 " data-hspacer="no_spacer"  data-halign="left" style="text-align:left"><div class="uvc-heading-spacer no_spacer" style="top"></div><div class="uvc-main-heading ult-responsive"  data-ultimate-target='.uvc-heading.ultimate-heading-465169fe346a57b39 h4'  data-responsive-json-new='{"font-size":"","line-height":""}' ><h4 style="font-weight:normal;">Subscribe for News Updates</h4></div></div><div class="vc_empty_space"   style="height: 16px" ><span class="vc_empty_space_inner"></span></div>

	<div class="wpb_text_column wpb_content_element " >
		<div class="wpb_wrapper">
			<div class="subscribe "><div class="subscribe-gf-form blue">
                <div class='gf_browser_unknown gform_wrapper' id='gform_wrapper_1' ><div id='gf_1' class='gform_anchor' tabindex='-1'></div><form method='post' enctype='multipart/form-data' target='gform_ajax_frame_1' id='gform_1'  action='/#gf_1'>
                        <div class='gform_body'><ul id='gform_fields_1' class='gform_fields top_label form_sublabel_below description_below'><li id='field_1_1'  class='gfield field_sublabel_below field_description_below gfield_visibility_visible' ><label class='gfield_label' for='input_1_1' >Email</label><div class='ginput_container ginput_container_email'>
                            <input name='input_1' id='input_1_1' type='text' value='' class='medium' tabindex='1'  placeholder='Enter email to subscribe'  aria-invalid="false" />
                        </div></li><li id='field_1_2'  class='gfield field_sublabel_below field_description_below gfield_visibility_visible' ><label class='gfield_label' for='input_1_2' >CAPTCHA</label><div id='input_1_2' class='ginput_container ginput_recaptcha' data-sitekey='6LfclTcUAAAAAPGFu_S4GUWvs2FmFrgaZcG8Ydql'  data-theme='light' data-tabindex='2'  data-badge=''></div></li><li id='field_1_3'  class='gfield gform_validation_container field_sublabel_below field_description_below gfield_visibility_visible' ><label class='gfield_label' for='input_1_3' >Email</label><div class='ginput_container'><input name='input_3' id='input_1_3' type='text' value='' /></div><div class='gfield_description' id='gfield_description__3'>This field is for validation purposes and should be left unchanged.</div></li>
                            </ul></div>
        <div class='gform_footer top_label'> <input type='image' src='/wp-content/themes/gsu-flex/images/subscribe-button.png' id='gform_submit_button_1' class='gform_button gform_image_button' alt='Submit' tabindex='3' onclick='if(window["gf_submitting_1"]){return false;}  window["gf_submitting_1"]=true;  ' onkeypress='if( event.keyCode == 13 ){ if(window["gf_submitting_1"]){return false;} window["gf_submitting_1"]=true;  jQuery("#gform_1").trigger("submit",[true]); }' /> <input type='hidden' name='gform_ajax' value='form_id=1&amp;title=&amp;description=&amp;tabindex=1' />
            <input type='hidden' class='gform_hidden' name='is_submit_1' value='1' />
            <input type='hidden' class='gform_hidden' name='gform_submit' value='1' />
            
            <input type='hidden' class='gform_hidden' name='gform_unique_id' value='' />
            <input type='hidden' class='gform_hidden' name='state_1' value='WyJbXSIsImRiYjRlYjViM2U4NGEyMGM4MDZhYWZhOWZlZTNkODhkIl0=' />
            <input type='hidden' class='gform_hidden' name='gform_target_page_number_1' id='gform_target_page_number_1' value='0' />
            <input type='hidden' class='gform_hidden' name='gform_source_page_number_1' id='gform_source_page_number_1' value='1' />
            <input type='hidden' name='gform_field_values' value='' />
            
        </div>
                        </form>
                        </div>
                <iframe style='display:none;width:0px;height:0px;' src='about:blank' name='gform_ajax_frame_1' id='gform_ajax_frame_1'>This iframe contains the logic required to handle Ajax powered Gravity Forms.</iframe>
                <script type='text/javascript'>jQuery(document).ready(function($){gformInitSpinner( 1, 'https://news.gsu.edu/wp-content/plugins/gravityforms/images/spinner.gif' );jQuery('#gform_ajax_frame_1').on('load',function(){var contents = jQuery(this).contents().find('*').html();var is_postback = contents.indexOf('GF_AJAX_POSTBACK') >= 0;if(!is_postback){return;}var form_content = jQuery(this).contents().find('#gform_wrapper_1');var is_confirmation = jQuery(this).contents().find('#gform_confirmation_wrapper_1').length > 0;var is_redirect = contents.indexOf('gformRedirect(){') >= 0;var is_form = form_content.length > 0 && ! is_redirect && ! is_confirmation;if(is_form){jQuery('#gform_wrapper_1').html(form_content.html());if(form_content.hasClass('gform_validation_error')){jQuery('#gform_wrapper_1').addClass('gform_validation_error');} else {jQuery('#gform_wrapper_1').removeClass('gform_validation_error');}setTimeout( function() { /* delay the scroll by 50 milliseconds to fix a bug in chrome */ jQuery(document).scrollTop(jQuery('#gform_wrapper_1').offset().top); }, 50 );if(window['gformInitDatepicker']) {gformInitDatepicker();}if(window['gformInitPriceFields']) {gformInitPriceFields();}var current_page = jQuery('#gform_source_page_number_1').val();gformInitSpinner( 1, 'https://news.gsu.edu/wp-content/plugins/gravityforms/images/spinner.gif' );jQuery(document).trigger('gform_page_loaded', [1, current_page]);window['gf_submitting_1'] = false;}else if(!is_redirect){var confirmation_content = jQuery(this).contents().find('.GF_AJAX_POSTBACK').html();if(!confirmation_content){confirmation_content = contents;}setTimeout(function(){jQuery('#gform_wrapper_1').replaceWith(confirmation_content);jQuery(document).scrollTop(jQuery('#gf_1').offset().top);jQuery(document).trigger('gform_confirmation_loaded', [1]);window['gf_submitting_1'] = false;}, 50);}else{jQuery('#gform_1').append(contents);if(window['gformRedirect']) {gformRedirect();}}jQuery(document).trigger('gform_post_render', [1, current_page]);} );} );</script></div></div>

		</div>
	</div>
<div class="vc_separator wpb_content_element vc_separator_align_center vc_sep_width_100 vc_sep_pos_align_center vc_separator_no_text vc_sep_color_grey" ><span class="vc_sep_holder vc_sep_holder_l"><span  class="vc_sep_line"></span></span><span class="vc_sep_holder vc_sep_holder_r"><span  class="vc_sep_line"></span></span>
</div><div id="ultimate-heading-258069fe346a5ca0a" class="uvc-heading ult-adjust-bottom-margin ultimate-heading-258069fe346a5ca0a uvc-3984 " data-hspacer="no_spacer"  data-halign="left" style="text-align:left"><div class="uvc-heading-spacer no_spacer" style="top"></div><div class="uvc-main-heading ult-responsive"  data-ultimate-target='.uvc-heading.ultimate-heading-258069fe346a5ca0a h4'  data-responsive-json-new='{"font-size":"","line-height":""}' ><h4 style="font-weight:normal;">Upcoming Events</h4></div></div><div class="vc_row wpb_row vc_inner vc_row-fluid"><div class="wpb_column vc_column_container vc_col-sm-12"><div class="vc_column-inner"><div class="wpb_wrapper">
	<div class="wpb_raw_code wpb_raw_js" >
		<div class="wpb_wrapper">
			<script type="text/javascript"
src="https://calendar.gsu.edu/widget/view?schools=georgiastate&days=365&num=5&picks=1&template=news-calendar"></script>
		</div>
	</div>
</div></div></div></div>
	<div class="wpb_text_column wpb_content_element  vc_custom_1619030632556" id="university-magazines">
		<div class="wpb_wrapper">
			
		</div>
	</div>
<span id="creative-link-wrap-6469" class="ult_main_cl ult-adjust-bottom-margin  " >
	 			<span class="ult_cl_link_3  ult_crlink" >
					<a  data-ultimate-target='#creative-link-wrap-6469 .ult_colorlink'  data-responsive-json-new='{"font-size":"","line-height":""}'  href="https://calendar.gsu.edu" target="_blank" rel="noopener"  class="ult_colorlink ult-responsive " style="font-weight:normal; "  data-textcolor="#0034a6" data-texthover="#0034a6"data-style="Style_3">
						
						<span data-hover="VIEW MORE UPCOMING EVENTS" style="color:#0034a6;;;" class="ult_btn10_span   ">VIEW MORE UPCOMING EVENTS</span>
						<span class="ult_link_btm3 " style="border-color:#cc0000;border-bottom-width:2px;border-style:solid;"></span>
					</a>
				</span>
			</span></div></div></div></div><div class="vc_row wpb_row vc_row-fluid vc_column-gap-35 vc_row-o-equal-height vc_row-flex"><div class="shadow-box wpb_column vc_column_container vc_col-sm-12 vc_col-has-fill"><div class="vc_column-inner vc_custom_1777648655677"><div class="wpb_wrapper"><div id="ultimate-heading-687369fe346a5dfec" class="uvc-heading ult-adjust-bottom-margin ultimate-heading-687369fe346a5dfec uvc-186 " data-hspacer="no_spacer"  data-halign="left" style="text-align:left"><div class="uvc-heading-spacer no_spacer" style="top"></div><div class="uvc-main-heading ult-responsive"  data-ultimate-target='.uvc-heading.ultimate-heading-687369fe346a5dfec h4'  data-responsive-json-new='{"font-size":"","line-height":""}' ><h4 style="font-weight:normal;">College/School News</h4></div></div><div class="vc_row wpb_row vc_inner vc_row-fluid vc_custom_1777651768608"><div class="wpb_column vc_column_container vc_col-sm-6"><div class="vc_column-inner"><div class="wpb_wrapper">
	<div class="wpb_text_column wpb_content_element " >
		<div class="wpb_wrapper">
			<div style="line-height: 34px;"><a href="/aysps-news/">Andrew Young School of Policy Studies</a><br />
<a href="/college-nursing-health-professions/" target="_blank" rel="noopener">Byrdine F. Lewis College of Nursing and Health Professions</a><br />
<a href="/arts-and-sciences-news/" target="_blank" rel="noopener">College of Arts and Sciences</a><br />
<a href="/cehd-news/" target="_blank" rel="noopener">College of Education &amp; Human Development</a><br />
<a href="/law-news/" target="_blank" rel="noopener">College of Law</a><br />
<a href="/college-arts-news/">College of The Arts</a></div>

		</div>
	</div>
</div></div></div><div class="wpb_column vc_column_container vc_col-sm-6"><div class="vc_column-inner"><div class="wpb_wrapper">
	<div class="wpb_text_column wpb_content_element " >
		<div class="wpb_wrapper">
			<div style="line-height: 34px;"><a href="/honors-news/">Honors College</a><br />
<a href="/ibms-news/" target="_blank" rel="noopener">Institute for Biomedical Sciences</a><br />
<a href="/robinson-news/" target="_blank" rel="noopener">J. Mack Robinson College of Business</a><br />
<a href="/perimeter-news/" target="_blank" rel="noopener">Perimeter College</a><br />
<a href="/public-health-news/" target="_blank" rel="noopener">School of Public Health</a></div>
<div style="line-height: 34px;"><a href="https://graduate.gsu.edu" target="_blank" rel="noopener">The Graduate School</a></div>

		</div>
	</div>
</div></div></div></div></div></div></div></div><div class="vc_row wpb_row vc_row-fluid feature-box-row vc_column-gap-10"><div class="shadow-box wpb_column vc_column_container vc_col-sm-12"><div class="vc_column-inner"><div class="wpb_wrapper"><div id="ultimate-heading-838369fe346a5f6a5" class="uvc-heading ult-adjust-bottom-margin ultimate-heading-838369fe346a5f6a5 uvc-3574 " data-hspacer="no_spacer"  data-halign="left" style="text-align:left"><div class="uvc-heading-spacer no_spacer" style="top"></div><div class="uvc-main-heading ult-responsive"  data-ultimate-target='.uvc-heading.ultimate-heading-838369fe346a5f6a5 h4'  data-responsive-json-new='{"font-size":"","line-height":""}' ><h4 style="font-weight:normal;">Featured Experts</h4></div></div><div class="vc_row wpb_row vc_inner vc_row-fluid vc_custom_1468336959421"><div class="wpb_column vc_column_container vc_col-sm-12 vc_col-md-9"><div class="vc_column-inner"><div class="wpb_wrapper"><!-- vc_grid start -->
<div class="vc_grid-container-wrapper vc_clearfix" >
	<div class="vc_grid-container vc_clearfix wpb_content_element vc_basic_grid featured-experts" data-initial-loading-animation="none" data-vc-grid-settings="{&quot;page_id&quot;:682,&quot;style&quot;:&quot;all&quot;,&quot;action&quot;:&quot;vc_get_vc_grid_data&quot;,&quot;shortcode_id&quot;:&quot;1777651571222-120cb959-2b8e-10&quot;,&quot;tag&quot;:&quot;vc_basic_grid&quot;}" data-vc-request="https://news.gsu.edu/wp-admin/admin-ajax.php" data-vc-post-id="682" data-vc-public-nonce="bc96b7788b">
		<style type="text/css" data-type="vc_shortcodes-custom-css">.vc_custom_1467238575415{margin-top: 10px !important;margin-right: 10px !important;margin-bottom: 10px !important;margin-left: 10px !important;background-color: rgba(38,46,60,0.75) !important;*background-color: rgb(38,46,60) !important;}.vc_custom_1467237032201{padding-right: 5px !important;padding-bottom: 15px !important;padding-left: 5px !important;}</style><div class="vc_grid vc_row vc_pageable-wrapper vc_hook_hover" data-vc-pageable-content="true"><div class="vc_pageable-slide-wrapper vc_clearfix" data-vc-grid-content="true"><div class="vc_grid-item vc_clearfix experts-home vc_col-sm-3 vc_grid-item-zone-c-bottom"><div class="vc_grid-item-mini vc_clearfix "><div class="vc_gitem-animated-block  vc_gitem-animate vc_gitem-animate-none" data-vc-animation="none"><div class="vc_gitem-zone vc_gitem-zone-a vc-gitem-zone-height-mode-auto vc-gitem-zone-height-mode-auto-1-1">
			<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-">
	<div class="wpb_single_image wpb_content_element vc_align_center">
		<figure class="wpb_wrapper vc_figure">
			<div class="vc_single_image-wrapper   vc_box_border_grey"><img width="300" height="300" src="https://news.gsu.edu/files/2024/04/fac-db-arun-rai.jpg" class="vc_single_image-img attachment-full" alt="Portrait of Arun Rai" /></div>
		</figure>
	</div>

</div></div>	</div>
</div>
<div class="vc_gitem-zone vc_gitem-zone-b vc_custom_1467238575415 vc-gitem-zone-height-mode-auto vc_gitem-is-link">
	<a href="https://news.gsu.edu/expert/arun-rai/" title="Arun Rai" class="vc_gitem-link vc-zone-link"></a>		<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-bottom"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align- vc_custom_1467237032201">	<div class="vc_gitem-post-meta-field-wpcf-expertdepts expert-hover vc_gitem-align-center"> Computer Information Systems
	</div>
<div class="vc_separator wpb_content_element vc_separator_align_center vc_sep_width_70 vc_sep_pos_align_center vc_separator_no_text vc_sep_color_white" ><span class="vc_sep_holder vc_sep_holder_l"><span  class="vc_sep_line"></span></span><span class="vc_sep_holder vc_sep_holder_r"><span  class="vc_sep_line"></span></span>
</div>	<div class="vc_gitem-post-meta-field-wpcf-schoolorcollege expert-hover vc_gitem-align-center"> J. Mack Robinson College of Business
	</div>
</div></div>	</div>
</div>
</div>
<div class="vc_gitem-zone vc_gitem-zone-c">
	<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-"><div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_title" ><h4 style="font-size: 1.8rem;color: #000000;line-height: 1.25;text-align: center;font-family:Lato;font-weight:400;font-style:normal" ><a href="https://news.gsu.edu/expert/arun-rai/" class="vc_gitem-link" title="Arun Rai">Arun Rai</a></h4></div>	<div class="vc_gitem-post-meta-field-wpcf-general-expertise expert-specializations vc_gitem-align-center"> Managing Generative AI; Digital Innovation and Solutions; Economic, Behavioral and Societal Impacts of Digital Technologies
	</div>
</div></div>	</div>
</div>
</div><div class="vc_clearfix"></div></div><div class="vc_grid-item vc_clearfix experts-home vc_col-sm-3 vc_grid-item-zone-c-bottom"><div class="vc_grid-item-mini vc_clearfix "><div class="vc_gitem-animated-block  vc_gitem-animate vc_gitem-animate-none" data-vc-animation="none"><div class="vc_gitem-zone vc_gitem-zone-a vc-gitem-zone-height-mode-auto vc-gitem-zone-height-mode-auto-1-1">
			<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-">
	<div class="wpb_single_image wpb_content_element vc_align_center">
		<figure class="wpb_wrapper vc_figure">
			<div class="vc_single_image-wrapper   vc_box_border_grey"><img width="300" height="300" src="https://news.gsu.edu/files/2024/04/Benjamin-Lawrence-300x300.jpg" class="vc_single_image-img attachment-full" alt="Portrait of Benjamin Lawrence." /></div>
		</figure>
	</div>

</div></div>	</div>
</div>
<div class="vc_gitem-zone vc_gitem-zone-b vc_custom_1467238575415 vc-gitem-zone-height-mode-auto vc_gitem-is-link">
	<a href="https://news.gsu.edu/expert/benjamin-lawrence/" title="Benjamin Lawrence" class="vc_gitem-link vc-zone-link"></a>		<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-bottom"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align- vc_custom_1467237032201">	<div class="vc_gitem-post-meta-field-wpcf-expertdepts expert-hover vc_gitem-align-center"> School of Hospitality Administration
	</div>
<div class="vc_separator wpb_content_element vc_separator_align_center vc_sep_width_70 vc_sep_pos_align_center vc_separator_no_text vc_sep_color_white" ><span class="vc_sep_holder vc_sep_holder_l"><span  class="vc_sep_line"></span></span><span class="vc_sep_holder vc_sep_holder_r"><span  class="vc_sep_line"></span></span>
</div>	<div class="vc_gitem-post-meta-field-wpcf-schoolorcollege expert-hover vc_gitem-align-center"> J. Mack Robinson College of Business
	</div>
</div></div>	</div>
</div>
</div>
<div class="vc_gitem-zone vc_gitem-zone-c">
	<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-"><div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_title" ><h4 style="font-size: 1.8rem;color: #000000;line-height: 1.25;text-align: center;font-family:Lato;font-weight:400;font-style:normal" ><a href="https://news.gsu.edu/expert/benjamin-lawrence/" class="vc_gitem-link" title="Benjamin Lawrence">Benjamin Lawrence</a></h4></div>	<div class="vc_gitem-post-meta-field-wpcf-general-expertise expert-specializations vc_gitem-align-center"> Franchise Ecosystem, Franchise Market Outlook and Trends
	</div>
</div></div>	</div>
</div>
</div><div class="vc_clearfix"></div></div><div class="vc_grid-item vc_clearfix experts-home vc_col-sm-3 vc_grid-item-zone-c-bottom"><div class="vc_grid-item-mini vc_clearfix "><div class="vc_gitem-animated-block  vc_gitem-animate vc_gitem-animate-none" data-vc-animation="none"><div class="vc_gitem-zone vc_gitem-zone-a vc-gitem-zone-height-mode-auto vc-gitem-zone-height-mode-auto-1-1">
			<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-">
	<div class="wpb_single_image wpb_content_element vc_align_center">
		<figure class="wpb_wrapper vc_figure">
			<div class="vc_single_image-wrapper   vc_box_border_grey"><img width="1463" height="2048" src="https://news.gsu.edu/files/2022/03/Karen_Minyard.jpg" class="vc_single_image-img attachment-full" alt="Karen Minyard" srcset="https://news.gsu.edu/files/2022/03/Karen_Minyard.jpg 1463w, https://news.gsu.edu/files/2022/03/Karen_Minyard-514x720.jpg 514w, https://news.gsu.edu/files/2022/03/Karen_Minyard-768x1075.jpg 768w, https://news.gsu.edu/files/2022/03/Karen_Minyard-804x1125.jpg 804w" sizes="(max-width: 1463px) 100vw, 1463px" /></div>
		</figure>
	</div>

</div></div>	</div>
</div>
<div class="vc_gitem-zone vc_gitem-zone-b vc_custom_1467238575415 vc-gitem-zone-height-mode-auto vc_gitem-is-link">
	<a href="https://news.gsu.edu/expert/karen-minyard/" title="Karen Minyard" class="vc_gitem-link vc-zone-link"></a>		<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-bottom"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align- vc_custom_1467237032201">	<div class="vc_gitem-post-meta-field-wpcf-expertdepts expert-hover vc_gitem-align-center"> Georgia Health Policy Center
	</div>
<div class="vc_separator wpb_content_element vc_separator_align_center vc_sep_width_70 vc_sep_pos_align_center vc_separator_no_text vc_sep_color_white" ><span class="vc_sep_holder vc_sep_holder_l"><span  class="vc_sep_line"></span></span><span class="vc_sep_holder vc_sep_holder_r"><span  class="vc_sep_line"></span></span>
</div>	<div class="vc_gitem-post-meta-field-wpcf-schoolorcollege expert-hover vc_gitem-align-center"> Andrew Young School of Policy Studies
	</div>
</div></div>	</div>
</div>
</div>
<div class="vc_gitem-zone vc_gitem-zone-c">
	<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-"><div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_title" ><h4 style="font-size: 1.8rem;color: #000000;line-height: 1.25;text-align: center;font-family:Lato;font-weight:400;font-style:normal" ><a href="https://news.gsu.edu/expert/karen-minyard/" class="vc_gitem-link" title="Karen Minyard">Karen Minyard</a></h4></div>	<div class="vc_gitem-post-meta-field-wpcf-general-expertise expert-specializations vc_gitem-align-center"> State Medicaid program Reform Design and Evaluation, Health-Related Social Policy Program Financing, Health and Health Care Financing
	</div>
</div></div>	</div>
</div>
</div><div class="vc_clearfix"></div></div><div class="vc_grid-item vc_clearfix experts-home vc_col-sm-3 vc_grid-item-zone-c-bottom"><div class="vc_grid-item-mini vc_clearfix "><div class="vc_gitem-animated-block  vc_gitem-animate vc_gitem-animate-none" data-vc-animation="none"><div class="vc_gitem-zone vc_gitem-zone-a vc-gitem-zone-height-mode-auto vc-gitem-zone-height-mode-auto-1-1">
			<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-">
	<div class="wpb_single_image wpb_content_element vc_align_center">
		<figure class="wpb_wrapper vc_figure">
			<div class="vc_single_image-wrapper   vc_box_border_grey"><img width="300" height="300" src="https://news.gsu.edu/files/2019/05/mia-belfast-photo-300x300.jpg" class="vc_single_image-img attachment-full" alt="Mia Bloom" /></div>
		</figure>
	</div>

</div></div>	</div>
</div>
<div class="vc_gitem-zone vc_gitem-zone-b vc_custom_1467238575415 vc-gitem-zone-height-mode-auto vc_gitem-is-link">
	<a href="https://news.gsu.edu/expert/mia-bloom/" title="Mia Bloom" class="vc_gitem-link vc-zone-link"></a>		<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-bottom"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align- vc_custom_1467237032201">	<div class="vc_gitem-post-meta-field-wpcf-expertdepts expert-hover vc_gitem-align-center"> Communication, Middle East Studies Center
	</div>
<div class="vc_separator wpb_content_element vc_separator_align_center vc_sep_width_70 vc_sep_pos_align_center vc_separator_no_text vc_sep_color_white" ><span class="vc_sep_holder vc_sep_holder_l"><span  class="vc_sep_line"></span></span><span class="vc_sep_holder vc_sep_holder_r"><span  class="vc_sep_line"></span></span>
</div>	<div class="vc_gitem-post-meta-field-wpcf-schoolorcollege expert-hover vc_gitem-align-center"> College of Arts and Sciences
	</div>
</div></div>	</div>
</div>
</div>
<div class="vc_gitem-zone vc_gitem-zone-c">
	<div class="vc_gitem-zone-mini">
		<div class="vc_gitem_row vc_row vc_gitem-row-position-top"><div class="vc_col-sm-12 vc_gitem-col vc_gitem-col-align-"><div class="vc_custom_heading vc_gitem-post-data vc_gitem-post-data-source-post_title" ><h4 style="font-size: 1.8rem;color: #000000;line-height: 1.25;text-align: center;font-family:Lato;font-weight:400;font-style:normal" ><a href="https://news.gsu.edu/expert/mia-bloom/" class="vc_gitem-link" title="Mia Bloom">Mia Bloom</a></h4></div>	<div class="vc_gitem-post-meta-field-wpcf-general-expertise expert-specializations vc_gitem-align-center"> Terrorism, Extremism, Child Soldiers
	</div>
</div></div>	</div>
</div>
</div><div class="vc_clearfix"></div></div></div></div>	</div>
</div><!-- vc_grid end -->
</div></div></div><div class="wpb_column vc_column_container vc_col-sm-12 vc_col-md-3 vc_col-has-fill"><div class="vc_column-inner vc_custom_1591979257539"><div class="wpb_wrapper">
	<div  class="wpb_single_image wpb_content_element vc_align_center  vc_custom_1558476788093">
		
		<figure class="wpb_wrapper vc_figure">
			<div class="vc_single_image-wrapper   vc_box_border_grey"><img class="vc_single_image-img " src="https://news.gsu.edu/files/2014/09/gsu-flame-transparentbg.png" width="25" height="25" alt="Georgia State Flame" title="gsu-flame-transparentbg" /></div>
		</figure>
	</div>

	<div class="wpb_text_column wpb_content_element  small" >
		<div class="wpb_wrapper">
			<p><span style="color: #ffffff;">Georgia State University has experts in every field. From health policy to geopolitical economics, we can help you.</span></p>

		</div>
	</div>
<div class=" ubtn-ctn-center "><a class="ubtn-link ult-adjust-bottom-margin ubtn-center ubtn-normal " href="/experts/" ><button type="button" id="ubtn-8490"  class="ubtn ult-adjust-bottom-margin ult-responsive ubtn-normal ubtn-no-hover-bg  none  ubtn-center   tooltip-69fe346a8bc01"  data-hover="" data-border-color="#ffffff" data-bg="" data-hover-bg="" data-border-hover="" data-shadow-hover="" data-shadow-click="none" data-shadow="" data-shd-shadow=""  data-ultimate-target='#ubtn-8490'  data-responsive-json-new='{"font-size":"","line-height":""}'  style="font-weight:normal;border-radius:0px;border-width:1px;border-color:#ffffff;border-style:solid;color: #ffffff;"><span class="ubtn-hover" style="background-color:"></span><span class="ubtn-data ubtn-text " >FIND AN EXPERT</span></button></a></div></div></div></div></div></div></div></div></div>
</div></div></article></main></div></div><footer class="site-footer" itemscope itemtype="https://schema.org/WPFooter"><div class="wrap">  <!-- !footer -->
   <div id="footer">
    <div id="sub-footer">
      	      </div>
    <div class="container"> 
      <!-- !#site-map -->
      <div id="site-map">
      	<div class="menu-site-map-container"><ul id="menu-site-map" class="menu"><li id="menu-item-6484788" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6484788 default-columns"><a target="_blank" href="https://www.gsu.edu/about/" itemprop="url">About</a>
<ul class="sub-menu">
	<li id="menu-item-6484652" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6484652 default-columns"><a href="https://www.gsu.edu/contact-georgia-state/" itemprop="url">Contact Georgia State</a></li>
	<li id="menu-item-4355129" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4355129 default-columns"><a target="_blank" href="https://employment.gsu.edu" itemprop="url">Employment</a></li>
	<li id="menu-item-4355130" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4355130 default-columns"><a target="_blank" href="https://gsu.policystat.com/" itemprop="url">University Policies</a></li>
	<li id="menu-item-4355131" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4355131 default-columns"><a target="_blank" href="https://president.gsu.edu" itemprop="url">Office of the President</a></li>
	<li id="menu-item-4732211" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4732211 default-columns"><a target="_blank" href="https://map.concept3d.com/?id=1108#!ct/0" itemprop="url">Campus Maps</a></li>
	<li id="menu-item-6484789" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-6484789 default-columns"><a href="https://www.gsu.edu/administration/" itemprop="url">Administration</a></li>
	<li id="menu-item-6499027" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-6499027 default-columns"><a target="_blank" href="https://news.gsu.edu" itemprop="url">News &#038; Events</a></li>
</ul>
</li>
<li id="menu-item-4732847" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4732847 default-columns"><a target="_blank" href="https://success.gsu.edu" itemprop="url">Student Success</a>
<ul class="sub-menu">
	<li id="menu-item-11" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-11 default-columns"><a title="Visit Admissions " target="_blank" href="https://admissions.gsu.edu" itemprop="url">Admissions</a></li>
	<li id="menu-item-9" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-9 default-columns"><a title="Explore Degrees and Majors" target="_blank" href="https://www.gsu.edu/program-cards/" itemprop="url">Degrees &#038; Majors</a></li>
	<li id="menu-item-4375460" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4375460 default-columns"><a title="Visit Graduate Programs" target="_blank" href="https://graduate.gsu.edu" itemprop="url">Graduate Programs</a></li>
	<li id="menu-item-6499025" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-6499025 default-columns"><a target="_blank" href="https://collegetocareer.gsu.edu" itemprop="url">College to Career</a></li>
	<li id="menu-item-52" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-52 default-columns"><a title="Visit Library" target="_blank" href="https://library.gsu.edu/" itemprop="url">University Library</a></li>
	<li id="menu-item-53" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-53 default-columns"><a target="_blank" href="https://lawlibrary.gsu.edu/" itemprop="url">Law Library</a></li>
	<li id="menu-item-4444144" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4444144 default-columns"><a title="Visit Bookstore" target="_blank" href="https://www.bkstr.com/georgiastatestore/home" itemprop="url">Bookstore</a></li>
</ul>
</li>
<li id="menu-item-4732848" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4732848 default-columns"><a href="#" itemprop="url">Connections</a>
<ul class="sub-menu">
	<li id="menu-item-4732849" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4732849 default-columns"><a title="GSU Magazine" target="_blank" href="https://magazine.gsu.edu" itemprop="url">Magazine</a></li>
	<li id="menu-item-65" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-65 default-columns"><a title="Alumni Resources" target="_blank" href="https://alumni.gsu.edu/" itemprop="url">Alumni</a></li>
	<li id="menu-item-68" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-68 default-columns"><a title="Donate to GSU" target="_blank" href="https://giving.gsu.edu" itemprop="url">Giving</a></li>
</ul>
</li>
<li id="menu-item-4732851" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-4732851 default-columns"><a href="#" itemprop="url">Arts &#038; Ideas</a>
<ul class="sub-menu">
	<li id="menu-item-4444173" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4444173 default-columns"><a title="Center for the Arts" target="_blank" href="https://rialto.gsu.edu/" itemprop="url">Rialto Center</a></li>
	<li id="menu-item-4727250" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4727250 default-columns"><a title="Listen to the Georgia State Radio Station" target="_blank" href="https://wrasfm.org/" itemprop="url">Album 88 (WRAS-FM)</a></li>
	<li id="menu-item-16" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-16 default-columns"><a title="University Research and Development" target="_blank" href="https://research.gsu.edu" itemprop="url">Research</a></li>
	<li id="menu-item-48" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-48 default-columns"><a title="University Research Services and Admin" target="_blank" href="https://ursa.research.gsu.edu" itemprop="url">URSA</a></li>
</ul>
</li>
<li id="menu-item-61" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-61 default-columns"><a target="_blank" href="https://www.georgiastatesports.com/" itemprop="url">Athletics</a>
<ul class="sub-menu">
	<li id="menu-item-63" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-63 default-columns"><a title="GSU Ticket Central" target="_blank" href="https://www.georgiastatesports.com/ViewArticle.dbml?DB_OEM_ID=12700&#038;ATCLID=211391891" itemprop="url">Tickets</a></li>
	<li id="menu-item-62" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-62 default-columns"><a title="GSU Athletics" target="_blank" href="https://www.georgiastatesports.com/" itemprop="url">Sports</a></li>
	<li id="menu-item-6482657" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-6482657 default-columns"><a target="_blank" href="https://georgiastatesports.com/sports/2020/12/8/ways-to-give" itemprop="url">Panther Athletic Club</a></li>
	<li id="menu-item-64" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-64 default-columns"><a title="Recruiting" target="_blank" href="https://georgiastatesports.com/sports/2006/9/18/610435.aspx" itemprop="url">Recruits</a></li>
	<li id="menu-item-4732853" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4732853 default-columns"><a title="GSU Athletic News" target="_blank" href="https://www.georgiastatesports.com/" itemprop="url">News</a></li>
</ul>
</li>
<li id="menu-item-54" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-54 default-columns"><a href="#" itemprop="url">Campus Life</a>
<ul class="sub-menu">
	<li id="menu-item-67" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-67 default-columns"><a title="Student Engagement" target="_blank" href="https://engagement.gsu.edu" itemprop="url">Student Engagement</a></li>
	<li id="menu-item-4433400" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4433400 default-columns"><a title="Counseling Center" target="_blank" href="https://counselingcenter.gsu.edu" itemprop="url">Counseling Center</a></li>
	<li id="menu-item-6505235" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-6505235 default-columns"><a target="_blank" href="https://health.gsu.edu" itemprop="url">Health Center</a></li>
	<li id="menu-item-55" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-55 default-columns"><a title="Housing" target="_blank" href="https://myhousing.gsu.edu" itemprop="url">Housing</a></li>
	<li id="menu-item-56" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-56 default-columns"><a title="Parking" target="_blank" href="https://parking.gsu.edu" itemprop="url">Parking</a></li>
	<li id="menu-item-4396961" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-4396961 default-columns"><a title="Dining" target="_blank" href="https://dining.gsu.edu" itemprop="url">Dining</a></li>
	<li id="menu-item-58" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-58 default-columns"><a title="Recreation" target="_blank" href="https://recreation.gsu.edu" itemprop="url">Recreation</a></li>
	<li id="menu-item-57" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-57 default-columns"><a title="Safety" target="_blank" href="https://safety.gsu.edu" itemprop="url">Safety</a></li>
</ul>
</li>
</ul></div>      </div>
      <!-- End #site-map -->
       <div class="footer-logo"><span></span></div>
	   <div class="footer-social">
	     <a aria-label="Go to Facebook" href="https://www.facebook.com/GeorgiaStateUniversity"  target="_blank"><span class="footer-facebook" role="img" aria-hidden="true"></span></a>
	     <a aria-label="Go to X" href="https://twitter.com/georgiastateu" target="_blank"><span class="footer-twitter" role="img" aria-hidden="true"></span></a>
	     <a aria-label="Go to LinkedIn" href="https://www.linkedin.com/edu/georgia-state-university-18163" target="_blank"><span class="footer-linkedin" role="img" aria-hidden="true"></span></a>
	     <a aria-label="Go to Instagram" href="https://instagram.com/georgiastateuniversity" target="_blank"><span class="footer-instagram" role="img" aria-hidden="true"></span></a>
	     <a aria-label="Go to YouTube" href="https://www.youtube.com/channel/UCpDtOZmGitIVXytVrjJL7LQ" target="_blank"><span class="footer-youtube" role="img" aria-hidden="true"></span></a>
</div>
	   <div class="footer-address">
	     <span>Georgia State University</span><span><a href="https://map.concept3d.com/?id=1108#!ce/27051?ct/0,27056,27057,27064,27066,27067,27070,27072,27073,27076,27078,27079,27082,27084,27085,63072,67808,67935,67936,67937,67950,67952,67953,67954,67955,67956,68437,68540,74628,40720,68538,67957,67980,67992,67993,67959,67966,67994,67995,67996,68033,67960,67967,67997,67998,67961,67968,67999,68000,67962,67969,67977,67984,67963,67970,67971,67985,68468,68926,68927,68928,27054?m/295232?s/?mc/33.75995174276261,-84.38365900000002?z/16?lvl/0?share">100 Auburn Ave NE Atlanta, GA 30303</a></span><span><a href="tel:+14044132000">404-413-2000</a></span>
	   </div>
	   <div class="footer-legal"><a href="https://www.gsu.edu/contact-georgia-state/" target="_blank">Contact Georgia State</a><a href="https://www.gsu.edu/legal-statement" target="_blank">View legal statement</a><a href="https://www.gsu.edu/privacy-notices/" target="_blank">Privacy Notices</a><a href="https://www.gsu.edu/state-authorization" target="_blank">State Authorization</a><a href="https://audit.gsu.edu/resources/georgia-state-ethics-hotline/" target="_blank">Ethics Hotline</a><a href="https://commkit.gsu.edu/website-management/support/" target="_blank">Website Feedback</a><span>©2026 Georgia State University</span></div>
    </div>
  </div>
        </div></footer></div><script type="text/html" id="tmpl-wp-playlist-current-item">
	<# if ( data.image ) { #>
	<img src="{{ data.thumb.src }}" alt="" />
	<# } #>
	<div class="wp-playlist-caption">
		<span class="wp-playlist-item-meta wp-playlist-item-title">&#8220;{{ data.title }}&#8221;</span>
		<# if ( data.meta.album ) { #><span class="wp-playlist-item-meta wp-playlist-item-album">{{ data.meta.album }}</span><# } #>
		<# if ( data.meta.artist ) { #><span class="wp-playlist-item-meta wp-playlist-item-artist">{{ data.meta.artist }}</span><# } #>
	</div>
</script>
<script type="text/html" id="tmpl-wp-playlist-item">
	<div class="wp-playlist-item">
		<a class="wp-playlist-caption" href="{{ data.src }}">
			{{ data.index ? ( data.index + '. ' ) : '' }}
			<# if ( data.caption ) { #>
				{{ data.caption }}
			<# } else { #>
				<span class="wp-playlist-item-title">&#8220;{{{ data.title }}}&#8221;</span>
				<# if ( data.artists && data.meta.artist ) { #>
				<span class="wp-playlist-item-artist"> &mdash; {{ data.meta.artist }}</span>
				<# } #>
			<# } #>
		</a>
		<# if ( data.meta.length_formatted ) { #>
		<div class="wp-playlist-item-length">{{ data.meta.length_formatted }}</div>
		<# } #>
	</div>
</script>

<script>
jQuery(document).ready(function($) {
    $(window).scroll(function () {
        if ($(window).scrollTop() > 0) {
						$('header').addClass('hide-sitetitle');
        }
        else{
						$('header').removeClass('hide-sitetitle');
        }
		});

});
</script>

<div id="views-extra-css-ie7" style="display:none;" aria-hidden="true">
<!--[if IE 7]><style>
.wpv-pagination { *zoom: 1; }
</style><![endif]-->
</div>

<script type="text/javascript">
jQuery( document ).ready( function( $ ) {
	var extra_css = $( "#views-extra-css" ) ? $( "#views-extra-css" ).text() : null;	if( extra_css ) {		$( 'head' ).append( '<style>' + extra_css + '</style>' );
		$( "#views-extra-css" ).remove();	}
	$( 'head' ).append( $( "#views-extra-css-ie7" ).html() );
	$( "#views-extra-css-ie7" ).remove();});
</script>
            <div class='asp_hidden_data' id="asp_hidden_data" style="display: none !important;">
                <svg style="position:absolute" height="0" width="0">
                    <filter id="aspblur">
                        <feGaussianBlur in="SourceGraphic" stdDeviation="4"/>
                    </filter>
                </svg>
                <svg style="position:absolute" height="0" width="0">
                    <filter id="no_aspblur"></filter>
                </svg>
            </div>
              <script type="text/javascript">

          var pfHeaderImgUrl = '';
          var pfHeaderTagline = '';
          var pfdisableClickToDel = '0';
          var pfImagesSize = 'full-size';
          var pfImageDisplayStyle = 'right';
          var pfEncodeImages = '0';
          var pfDisableEmail = '0';
          var pfDisablePDF = '0';
          var pfDisablePrint = '0';
          var pfCustomCSS = '';
          var pfPlatform = 'Wordpress';
      (function() {
            var e = document.createElement('script'); e.type="text/javascript";
            e.src = 'https://cdn.printfriendly.com/printfriendly.js';
            document.getElementsByTagName('head')[0].appendChild(e);
        })();
      </script>
<link rel='stylesheet' id='prettyphoto-css'  href='https://news.gsu.edu/wp-content/plugins/js_composer/assets/lib/prettyphoto/css/prettyPhoto.min.css?ver=5.7' type='text/css' media='all' />
<link rel='stylesheet' id='vc_pageable_owl-carousel-css-css'  href='https://news.gsu.edu/wp-content/plugins/js_composer/assets/lib/owl-carousel2-dist/assets/owl.min.css?ver=5.7' type='text/css' media='all' />
<link rel='stylesheet' id='animate-css-css'  href='https://news.gsu.edu/wp-content/plugins/js_composer/assets/lib/bower/animate-css/animate.min.css?ver=5.7' type='text/css' media='all' />
<link rel='stylesheet' id='gforms_reset_css-css'  href='https://news.gsu.edu/wp-content/plugins/gravityforms/css/formreset.min.css?ver=2.4.9' type='text/css' media='all' />
<link rel='stylesheet' id='gforms_formsmain_css-css'  href='https://news.gsu.edu/wp-content/plugins/gravityforms/css/formsmain.min.css?ver=2.4.9' type='text/css' media='all' />
<link rel='stylesheet' id='gforms_ready_class_css-css'  href='https://news.gsu.edu/wp-content/plugins/gravityforms/css/readyclass.min.css?ver=2.4.9' type='text/css' media='all' />
<link rel='stylesheet' id='gforms_browsers_css-css'  href='https://news.gsu.edu/wp-content/plugins/gravityforms/css/browsers.min.css?ver=2.4.9' type='text/css' media='all' />
<style type="text/css">@font-face {font-family:Lato;font-style:normal;font-weight:100;src:url(/cf-fonts/s/lato/5.0.18/latin/100/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:100;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/100/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:italic;font-weight:100;src:url(/cf-fonts/s/lato/5.0.18/latin/100/italic.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Lato;font-style:italic;font-weight:100;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/100/italic.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:300;src:url(/cf-fonts/s/lato/5.0.18/latin/300/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:300;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/300/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:italic;font-weight:300;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/300/italic.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:italic;font-weight:300;src:url(/cf-fonts/s/lato/5.0.18/latin/300/italic.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:400;src:url(/cf-fonts/s/lato/5.0.18/latin/400/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:400;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/400/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:italic;font-weight:400;src:url(/cf-fonts/s/lato/5.0.18/latin/400/italic.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Lato;font-style:italic;font-weight:400;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/400/italic.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:700;src:url(/cf-fonts/s/lato/5.0.18/latin/700/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:700;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/700/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:italic;font-weight:700;src:url(/cf-fonts/s/lato/5.0.18/latin/700/italic.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Lato;font-style:italic;font-weight:700;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/700/italic.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:900;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/900/normal.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:normal;font-weight:900;src:url(/cf-fonts/s/lato/5.0.18/latin/900/normal.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}@font-face {font-family:Lato;font-style:italic;font-weight:900;src:url(/cf-fonts/s/lato/5.0.18/latin-ext/900/italic.woff2);unicode-range:U+0100-02AF,U+0304,U+0308,U+0329,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20CF,U+2113,U+2C60-2C7F,U+A720-A7FF;font-display:swap;}@font-face {font-family:Lato;font-style:italic;font-weight:900;src:url(/cf-fonts/s/lato/5.0.18/latin/900/italic.woff2);unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;font-display:swap;}</style>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/revslider-typewriter-addon/public/assets/js/revolution.addon.typewriter.min.js?ver=1.0.3'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/themes/genesis/lib/js/skip-links.js?ver=2.5.3'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/themes/gsu-flex-2.1-magazine/javascript/modernizr-mq.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/themes/gsu-flex-2.1-magazine/javascript/slick/slick.min.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/themes/gsu-flex-2.1-magazine/javascript/jquery.columnizer.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/themes/gsu-flex-2.1-magazine/javascript/readmore/readmore.min.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/themes/gsu-flex-2.1-magazine/javascript/jquery.helperFunctions.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/themes/gsu-flex-2.1-magazine/javascript/accordion-menu.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/themes/gsu-flex-2.1-magazine/javascript/gsu-flex.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/types/vendor/toolset/toolset-common/res/lib/bootstrap/js/bootstrap.min.js?ver=3.3.7'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/ajax-search-pro/js/nomin/photostack.js?ver=xaNzoc'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/ajax-search-pro/js/min/chosen.jquery.min.js?ver=xaNzoc'></script>
<script type='text/javascript'>
/* <![CDATA[ */
var ajaxsearchpro = {"ajaxurl":"https:\/\/news.gsu.edu\/wp-admin\/admin-ajax.php","backend_ajaxurl":"https:\/\/news.gsu.edu\/wp-admin\/admin-ajax.php","js_scope":"jQuery"};
var ASP = {"ajaxurl":"https:\/\/news.gsu.edu\/wp-admin\/admin-ajax.php","backend_ajaxurl":"https:\/\/news.gsu.edu\/wp-admin\/admin-ajax.php","js_scope":"jQuery","asp_url":"https:\/\/news.gsu.edu\/wp-content\/plugins\/ajax-search-pro\/","upload_url":"https:\/\/news.gsu.edu\/files\/asp_upload\/","detect_ajax":"0","media_query":"xaNzoc","version":"4984","scrollbar":"1","css_loaded":"1","js_retain_popstate":"0","fix_duplicates":"1"};
/* ]]> */
</script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/ajax-search-pro/js/min/jquery.ajaxsearchpro-noui-isotope.min.js?ver=xaNzoc'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/jquery/ui/datepicker.min.js?ver=1.11.4'></script>
<script type='text/javascript'>
jQuery(document).ready(function(jQuery){jQuery.datepicker.setDefaults({"closeText":"Close","currentText":"Today","monthNames":["January","February","March","April","May","June","July","August","September","October","November","December"],"monthNamesShort":["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],"nextText":"Next","prevText":"Previous","dayNames":["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"],"dayNamesShort":["Sun","Mon","Tue","Wed","Thu","Fri","Sat"],"dayNamesMin":["S","M","T","W","T","F","S"],"dateFormat":"MM d, yy","firstDay":0,"isRTL":false});});
</script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/essb-functional-buttons-pack/assets/clipboard.min.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/essb-functional-buttons-pack/assets/qrcode.min.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/easy-social-share-buttons3/assets/js/essb-core.min.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/page-links-to/js/new-tab.min.js?ver=3.0.0-beta.1'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/mediaelement/wp-mediaelement.min.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/underscore.min.js?ver=1.8.3'></script>
<script type='text/javascript'>
/* <![CDATA[ */
var _wpUtilSettings = {"ajax":{"url":"\/wp-admin\/admin-ajax.php"}};
/* ]]> */
</script>
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/wp-util.min.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/backbone.min.js?ver=1.2.3'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/mediaelement/wp-playlist.min.js?ver=4.9.8'></script>
<script type='text/javascript'>
/* <![CDATA[ */
var wpv_pagination_local = {"front_ajaxurl":"https:\/\/news.gsu.edu\/wp-admin\/admin-ajax.php","calendar_image":"https:\/\/news.gsu.edu\/wp-content\/plugins\/wp-views\/embedded\/res\/img\/calendar.gif","calendar_text":"Select date","datepicker_min_date":null,"datepicker_max_date":null,"datepicker_min_year":"1582","datepicker_max_year":"3000","resize_debounce_tolerance":"100","datepicker_style_url":"https:\/\/news.gsu.edu\/wp-content\/plugins\/types\/vendor\/toolset\/toolset-common\/toolset-forms\/css\/wpt-jquery-ui\/jquery-ui-1.11.4.custom.css","wpmlLang":""};
/* ]]> */
</script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/wp-views/embedded/res/js/wpv-pagination-embedded.js?ver=2.8.2'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/imagesloaded.min.js?ver=3.2.0'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/masonry.min.js?ver=3.3.2'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/jquery/ui/widget.min.js?ver=1.11.4'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/jquery/ui/mouse.min.js?ver=1.11.4'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/jquery/ui/slider.min.js?ver=1.11.4'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-includes/js/wp-embed.min.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/js_composer/assets/js/dist/js_composer_front.min.js?ver=5.7'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/js_composer/assets/lib/prettyphoto/js/jquery.prettyPhoto.min.js?ver=5.7'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/js_composer/assets/lib/owl-carousel2-dist/owl.carousel.min.js?ver=5.7'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/js_composer/assets/lib/bower/imagesloaded/imagesloaded.pkgd.min.js?ver=4.9.8'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/js_composer/assets/lib/waypoints/waypoints.min.js?ver=5.7'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/js_composer/assets/js/dist/vc_grid.min.js?ver=5.7'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/js_composer/assets/lib/bower/twbs-pagination/jquery.twbsPagination.min.js?ver=5.7'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/gravityforms/js/jquery.json.min.js?ver=2.4.9'></script>
<script type='text/javascript'>
/* <![CDATA[ */
var gf_global = {"gf_currency_config":{"name":"U.S. Dollar","symbol_left":"$","symbol_right":"","symbol_padding":"","thousand_separator":",","decimal_separator":".","decimals":2},"base_url":"https:\/\/news.gsu.edu\/wp-content\/plugins\/gravityforms","number_formats":[],"spinnerUrl":"https:\/\/news.gsu.edu\/wp-content\/plugins\/gravityforms\/images\/spinner.gif"};
/* ]]> */
</script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/gravityforms/js/gravityforms.min.js?ver=2.4.9'></script>
<script type='text/javascript' src='https://news.gsu.edu/wp-content/plugins/gravityforms/js/placeholders.jquery.min.js?ver=2.4.9'></script>
<script type='text/javascript' src='https://www.google.com/recaptcha/api.js?hl=en&#038;render=explicit&#038;ver=4.9.8'></script>
<script type='text/javascript'> jQuery(document).bind('gform_post_render', function(event, formId, currentPage){if(formId == 1) {if(typeof Placeholders != 'undefined'){
                        Placeholders.enable();
                    }} } );jQuery(document).bind('gform_post_conditional_logic', function(event, formId, fields, isInit){} );</script><script type='text/javascript'> jQuery(document).ready(function(){jQuery(document).trigger('gform_post_render', [1, 1]) } ); </script>		<script type="text/javascript">
			( function( $ ) {
				$( document ).bind( 'gform_post_render', function() {
					var gfRecaptchaPoller = setInterval( function() {
						if( ! window.grecaptcha || ! window.grecaptcha.render ) {
							return;
						}
						renderRecaptcha();
						clearInterval( gfRecaptchaPoller );
					}, 100 );
				} );
			} )( jQuery );
		</script>

		<script type="text/javascript">
	function essbf_bookmark_page(e) {
		e.preventDefault();
		var bookmarkURL = window.location.href;
    	var bookmarkTitle = document.title;

    	if ('addToHomescreen' in window && addToHomescreen.isCompatible) {
		      // Mobile browsers
      		addToHomescreen({ autostart: false, startDelay: 0 }).show(true);
    	} else if (window.sidebar && window.sidebar.addPanel) {
		      // Firefox <=22
      		window.sidebar.addPanel(bookmarkTitle, bookmarkURL, '');
    	} else if ((window.sidebar && /Firefox/i.test(navigator.userAgent)) || (window.opera && window.print)) {
		      // Firefox 23+ and Opera <=14
      		jQuery(this).attr({
        		href: bookmarkURL,
        		title: bookmarkTitle,
        		rel: 'sidebar'
      		}).off(e);
      		return true;
    	} else if (window.external && ('AddFavorite' in window.external)) {
      		// IE Favorites
      		window.external.AddFavorite(bookmarkURL, bookmarkTitle);
    	} else {
      		// Other browsers (mainly WebKit & Blink - Safari, Chrome, Opera 15+)
     		 alert('Press ' + (/Mac/i.test(navigator.userAgent) ? 'Cmd' : 'Ctrl') + '+D to bookmark this page.');
    	}

   		return false;
	}
	
	function essbfp_copylink(instance) {
	    return false;
	}	
	
	function essbfp_qrcode(instance, url) {
		var element = jQuery("#essb-qr-"+instance);
		
		if (!jQuery(element).hasClass("generated")) {
			var qrcode = new QRCode("essb-qr-"+instance);
			qrcode.makeCode(url);
			jQuery(element).addClass("generated");
		}
		
		setTimeout(function() {
			var qrContent = document.getElementById("essb-qr-"+instance).innerHTML;
			essbQRCodeOpenPopup(qrContent);
		}, 1);
		return false;
	}
	
	function essbQRCodeOpenPopup(content){ 
    winpops = window.open(
        '',
        'QR Code',
        'fullscreen=no, toolbar=yes, status=yes, menubar=yes, scrollbars=yes, resizable=yes, directories=yes, location=yes, width=500, height=400, left=100, top=100, screenX=100, screenY=100');
     
    // Write the content to the popup
    winpops.document.write('<style type="text/css">#content div { font-family: sans-serif; font-weight: bold; text-align:center; margin: 20px 0; } #content img { margin: 0 auto; }</style><div id="content" style="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen-Sans,Ubuntu,Cantarell,Helvetica Neue,sans-serif">'+content+'</div>');
} 
	
	jQuery( document ).ready(function() {
		try {
	    	new Clipboard('.essb_link_copylink a', {
	    		text: function(trigger) {
	    			var element = trigger.parentElement.parentElement.parentElement;
	    			var url = jQuery(element).attr("data-essb-url") || "";
	    			console.log("closest URL = " +url);
	        		return url;
	    		}
			});
		}
		catch (e) {
		}
	});
	</script><link rel="stylesheet" id="essb-cct-style"  href="https://news.gsu.edu/wp-content/plugins/easy-social-share-buttons3/lib/modules/click-to-tweet/assets/css/styles.min.css" type="text/css" media="all" /><script type="text/javascript"></script><script type="text/javascript">window.NREUM||(NREUM={});NREUM.info={"beacon":"bam.nr-data.net","licenseKey":"1be6db1351","applicationID":"318032455","transactionName":"YlVUYkNVWEUDBUFbVlsfY0RYG19YBgNNHEldQA==","queueTime":0,"applicationTime":322,"atts":"ThJXFAtPS0s=","errorBeacon":"bam.nr-data.net","agent":""}</script><script>(function(){function c(){var b=a.contentDocument||a.contentWindow.document;if(b){var d=b.createElement('script');d.innerHTML="window.__CF$cv$params={r:'9f985893c8863c79',t:'MTc3ODQwOTgzOA=='};var a=document.createElement('script');a.src='/cdn-cgi/challenge-platform/scripts/jsd/main.js';document.getElementsByTagName('head')[0].appendChild(a);";b.getElementsByTagName('head')[0].appendChild(d)}}if(document.body){var a=document.createElement('iframe');a.height=1;a.width=1;a.style.position='absolute';a.style.top=0;a.style.left=0;a.style.border='none';a.style.visibility='hidden';document.body.appendChild(a);if('loading'!==document.readyState)c();else if(window.addEventListener)document.addEventListener('DOMContentLoaded',c);else{var e=document.onreadystatechange||function(){};document.onreadystatechange=function(b){e(b);'loading'!==document.readyState&&(document.onreadystatechange=e,c())}}}})();</script></body></html>

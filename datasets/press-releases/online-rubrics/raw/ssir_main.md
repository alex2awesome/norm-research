<!-- URL: https://ssir.org/ -->
<!-- HTTP: 200 SIZE:   194729 -->
<!DOCTYPE html>
<html lang="en-us">
    <head>
        <meta charset="UTF-8" /><script type="text/javascript">(window.NREUM||(NREUM={})).init={privacy:{cookies_enabled:true},ajax:{deny_list:["bam.nr-data.net"]},feature_flags:["soft_nav"],distributed_tracing:{enabled:true}};(window.NREUM||(NREUM={})).loader_config={agentID:"1103296721",accountID:"4260910",trustKey:"4260910",xpid:"UAQBUV9SCBABVFdWAQYDVFIJ",licenseKey:"NRJS-59f45596d67da7d3e71",applicationID:"1064674268",browserID:"1103296721"};;/*! For license information please see nr-loader-spa-1.314.0.min.js.LICENSE.txt */
(()=>{var e,t,r={384:(e,t,r)=>{"use strict";r.d(t,{NT:()=>a,Zm:()=>c,bQ:()=>u,dV:()=>d,pV:()=>l});var n=r(6154),i=r(1863),s=r(944),o=r(1910);const a={beacon:"bam.nr-data.net",errorBeacon:"bam.nr-data.net"};function c(){return n.gm.NREUM||(n.gm.NREUM={}),void 0===n.gm.newrelic&&(n.gm.newrelic=n.gm.NREUM),n.gm.NREUM}function d(){let e=c();return e.o||(e.o={ST:n.gm.setTimeout,SI:n.gm.setImmediate||n.gm.setInterval,CT:n.gm.clearTimeout,XHR:n.gm.XMLHttpRequest,REQ:n.gm.Request,EV:n.gm.Event,PR:n.gm.Promise,MO:n.gm.MutationObserver,FETCH:n.gm.fetch,WS:n.gm.WebSocket},(0,o.i)(...Object.values(e.o))),e}function u(e,t){let r=c();r.initializedAgents??={},t.initializedAt={ms:(0,i.t)(),date:new Date},r.initializedAgents[e]=t,2===Object.keys(r.initializedAgents).length&&(0,s.R)(69)}function l(){return function(){let e=c();const t=e.info||{};e.info={beacon:a.beacon,errorBeacon:a.errorBeacon,...t}}(),function(){let e=c();const t=e.init||{};e.init={...t}}(),d(),function(){let e=c();const t=e.loader_config||{};e.loader_config={...t}}(),c()}},782:(e,t,r)=>{"use strict";r.d(t,{T:()=>n});const n=r(860).K7.pageViewTiming},860:(e,t,r)=>{"use strict";r.d(t,{$J:()=>u,K7:()=>c,P3:()=>d,XX:()=>i,Yy:()=>a,df:()=>s,qY:()=>n,v4:()=>o});const n="events",i="jserrors",s="browser/blobs",o="rum",a="browser/logs",c={ajax:"ajax",genericEvents:"generic_events",jserrors:i,logging:"logging",metrics:"metrics",pageAction:"page_action",pageViewEvent:"page_view_event",pageViewTiming:"page_view_timing",sessionReplay:"session_replay",sessionTrace:"session_trace",softNav:"soft_navigations"},d={[c.pageViewEvent]:1,[c.pageViewTiming]:2,[c.metrics]:3,[c.jserrors]:4,[c.softNav]:5,[c.ajax]:6,[c.sessionTrace]:7,[c.sessionReplay]:8,[c.logging]:9,[c.genericEvents]:10},u={[c.pageViewEvent]:o,[c.pageViewTiming]:n,[c.ajax]:n,[c.softNav]:n,[c.metrics]:i,[c.jserrors]:i,[c.sessionTrace]:s,[c.sessionReplay]:s,[c.logging]:a,[c.genericEvents]:"ins"}},944:(e,t,r)=>{"use strict";r.d(t,{R:()=>i});var n=r(3241);function i(e,t){"function"==typeof console.debug&&(console.debug("New Relic Warning: https://github.com/newrelic/newrelic-browser-agent/blob/main/docs/warning-codes.md#".concat(e),t),(0,n.W)({drained:null,type:"data",name:"warn",feature:"warn",data:{code:e,secondary:t}}))}},993:(e,t,r)=>{"use strict";r.d(t,{A$:()=>s,ET:()=>o,TZ:()=>a,p_:()=>i});var n=r(860);const i={ERROR:"ERROR",WARN:"WARN",INFO:"INFO",DEBUG:"DEBUG",TRACE:"TRACE"},s={OFF:0,ERROR:1,WARN:2,INFO:3,DEBUG:4,TRACE:5},o="log",a=n.K7.logging},1687:(e,t,r)=>{"use strict";r.d(t,{Ak:()=>a,Ze:()=>d,x3:()=>c});var n=r(3241),i=r(3606),s=r(860),o=r(2646);function a(e,t){if(!e)return;const r={staged:!1,priority:s.P3[t]||0};e.runtime.drainRegistry.get(t)||e.runtime.drainRegistry.set(t,r)}function c(e,t){if(!e)return;const r=e.runtime.drainRegistry;r&&(r.get(t)&&r.delete(t),l(e,t,!1),r.size&&u(e))}function d(e,t="feature",r=!1){if(e){if(!e.runtime.drainRegistry.get(t)||r)return l(e,t);e.runtime.drainRegistry.get(t).staged=!0,u(e)}}function u(e){if(!e)return;const t=Array.from(e.runtime.drainRegistry);t.every(([e,t])=>t.staged)&&(t.sort((e,t)=>e[1].priority-t[1].priority),t.forEach(([t])=>{e.runtime.drainRegistry.delete(t),l(e,t)}))}function l(e,t,r=!0){if(!e)return;const s=e.ee,a=i.i.handlers;if(s&&!s.aborted&&s.backlog&&a){if((0,n.W)({type:"lifecycle",name:"drain",feature:t}),r){const e=s.backlog[t],r=a[t];if(r){for(let t=0;e&&t<e.length;++t)f(e[t],r);Object.entries(r).forEach(([e,t])=>{Object.values(t||{}).forEach(t=>{t[0]?.on&&t[0].context()instanceof o.y&&!t[0].listeners(e).includes(t[1])&&t[0].on(e,t[1])})})}}s.isolatedBacklog||delete a[t],s.backlog[t]=null,s.emit("drain-"+t,[])}}function f(e,t){var r=e[1];Object.values(t[r]||{}).forEach(t=>{var r=e[0];if(t[0]===r){var n=t[1],i=e[3],s=e[2];n.apply(i,s)}})}},1738:(e,t,r)=>{"use strict";r.d(t,{U:()=>f,Y:()=>l});var n=r(3241),i=r(9908),s=r(1863),o=r(944),a=r(3969),c=r(8362),d=r(860),u=r(4261);function l(e,t,r,s){const l=s||r;!l||l[e]&&l[e]!==c.d.prototype[e]||(l[e]=function(){(0,i.p)(a.xV,["API/"+e+"/called"],void 0,d.K7.metrics,r.ee),(0,n.W)({drained:!!r.runtime?.activatedFeatures,type:"data",name:"api",feature:u.Pl+e,data:{}});try{return t.apply(this,arguments)}catch(e){(0,o.R)(23,e)}})}function f(e,t,r,n,o){const a=e.info;null===r?delete a.jsAttributes[t]:a.jsAttributes[t]=r,(o||null===r)&&(0,i.p)(u.Pl+n,[(0,s.t)(),t,r],void 0,"session",e.ee)}},1741:(e,t,r)=>{"use strict";r.d(t,{W:()=>s});var n=r(944),i=r(4261);class s{#e(e,...t){if(this[e]!==s.prototype[e])return this[e](...t);(0,n.R)(35,e)}addPageAction(e,t){return this.#e(i.hG,e,t)}register(e){return this.#e(i.eY,e)}recordCustomEvent(e,t){return this.#e(i.fF,e,t)}setPageViewName(e,t){return this.#e(i.Fw,e,t)}setCustomAttribute(e,t,r){return this.#e(i.cD,e,t,r)}noticeError(e,t){return this.#e(i.o5,e,t)}setUserId(e,t=!1){return this.#e(i.Dl,e,t)}setApplicationVersion(e){return this.#e(i.nb,e)}setErrorHandler(e){return this.#e(i.bt,e)}addRelease(e,t){return this.#e(i.k6,e,t)}log(e,t){return this.#e(i.$9,e,t)}start(){return this.#e(i.d3)}finished(e){return this.#e(i.BL,e)}recordReplay(){return this.#e(i.CH)}pauseReplay(){return this.#e(i.Tb)}addToTrace(e){return this.#e(i.U2,e)}setCurrentRouteName(e){return this.#e(i.PA,e)}interaction(e){return this.#e(i.dT,e)}wrapLogger(e,t,r){return this.#e(i.Wb,e,t,r)}measure(e,t){return this.#e(i.V1,e,t)}consent(e){return this.#e(i.Pv,e)}}},1863:(e,t,r)=>{"use strict";function n(){return Math.floor(performance.now())}r.d(t,{t:()=>n})},1910:(e,t,r)=>{"use strict";r.d(t,{i:()=>s});var n=r(944);const i=new Map;function s(...e){return e.every(e=>{if(i.has(e))return i.get(e);const t="function"==typeof e?e.toString():"",r=t.includes("[native code]"),s=t.includes("nrWrapper");return r||s||(0,n.R)(64,e?.name||t),i.set(e,r),r})}},2555:(e,t,r)=>{"use strict";r.d(t,{D:()=>a,f:()=>o});var n=r(384),i=r(8122);const s={beacon:n.NT.beacon,errorBeacon:n.NT.errorBeacon,licenseKey:void 0,applicationID:void 0,sa:void 0,queueTime:void 0,applicationTime:void 0,ttGuid:void 0,user:void 0,account:void 0,product:void 0,extra:void 0,jsAttributes:{},userAttributes:void 0,atts:void 0,transactionName:void 0,tNamePlain:void 0};function o(e){try{return!!e.licenseKey&&!!e.errorBeacon&&!!e.applicationID}catch(e){return!1}}const a=e=>(0,i.a)(e,s)},2614:(e,t,r)=>{"use strict";r.d(t,{BB:()=>o,H3:()=>n,g:()=>d,iL:()=>c,tS:()=>a,uh:()=>i,wk:()=>s});const n="NRBA",i="SESSION",s=144e5,o=18e5,a={STARTED:"session-started",PAUSE:"session-pause",RESET:"session-reset",RESUME:"session-resume",UPDATE:"session-update"},c={SAME_TAB:"same-tab",CROSS_TAB:"cross-tab"},d={OFF:0,FULL:1,ERROR:2}},2646:(e,t,r)=>{"use strict";r.d(t,{y:()=>n});class n{constructor(e){this.contextId=e}}},2843:(e,t,r)=>{"use strict";r.d(t,{G:()=>s,u:()=>i});var n=r(3878);function i(e,t=!1,r,i){(0,n.DD)("visibilitychange",function(){if(t)return void("hidden"===document.visibilityState&&e());e(document.visibilityState)},r,i)}function s(e,t,r){(0,n.sp)("pagehide",e,t,r)}},3241:(e,t,r)=>{"use strict";r.d(t,{W:()=>s});var n=r(6154);const i="newrelic";function s(e={}){try{n.gm.dispatchEvent(new CustomEvent(i,{detail:e}))}catch(e){}}},3304:(e,t,r)=>{"use strict";r.d(t,{A:()=>s});var n=r(7836);const i=()=>{const e=new WeakSet;return(t,r)=>{if("object"==typeof r&&null!==r){if(e.has(r))return;e.add(r)}return r}};function s(e){try{return JSON.stringify(e,i())??""}catch(e){try{n.ee.emit("internal-error",[e])}catch(e){}return""}}},3333:(e,t,r)=>{"use strict";r.d(t,{$v:()=>u,TZ:()=>n,Xh:()=>c,Zp:()=>i,kd:()=>d,mq:()=>a,nf:()=>o,qN:()=>s});const n=r(860).K7.genericEvents,i=["auxclick","click","copy","keydown","paste","scrollend"],s=["focus","blur"],o=4,a=1e3,c=2e3,d=["PageAction","UserAction","BrowserPerformance"],u={RESOURCES:"experimental.resources",REGISTER:"register"}},3434:(e,t,r)=>{"use strict";r.d(t,{Jt:()=>o,YM:()=>u});var n=r(7836),i=r(5607),s=r(5732);const o="nr@original:".concat(i.W),a=50;var c=Object.prototype.hasOwnProperty,d=!1;function u(e,t,r){return e||(e=n.ee),i.inPlace=function(e,t,r,n,s,o){r||(r="");const a="-"===r.charAt(0);for(let c=0;c<t.length;c++){const d=t[c],u=e[d];f(u)||(e[d]=i(u,a?d+r:r,n,d,s,o))}},i.flag=o,i;function i(t,n,i,d,h,p){return f(t)?t:(n||(n=""),nrWrapper[o]=t,function(e,t,r){if(Object.defineProperty&&Object.keys)try{return Object.keys(e).forEach(function(r){Object.defineProperty(t,r,{get:function(){return e[r]},set:function(t){return e[r]=t,t}})}),t}catch(e){l([e],r)}for(var n in e)c.call(e,n)&&(t[n]=e[n])}(t,nrWrapper,e),nrWrapper);function nrWrapper(){var o,c,f,g;let m,v;try{c=this,o=[...arguments],v=p?(0,s.$5)(r):[void 0],f="function"==typeof i?i(o,c):i||{}}catch(t){l([t,"",[o,c,d],f],e)}u(n+"start",[o,c,d,v],f,h);const y=performance.now();let b;try{return g=t.apply(c,o),b=performance.now(),g}catch(e){throw b=performance.now(),u(n+"err",[o,c,e,v],f,h),m=e,m}finally{const e=b-y,t={start:y,end:b,duration:e,isLongTask:e>=a,methodName:d,thrownError:m};t.isLongTask&&u("long-task",[t,c,v],f,h),u(n+"end",[o,c,g,v],f,h)}}}function u(r,n,i,s){if(!d||t){var o=d;d=!0;try{e.emit(r,n,i,t,s)}catch(t){l([t,r,n,i],e)}d=o}}}function l(e,t){t||(t=n.ee);try{t.emit("internal-error",e)}catch(e){}}function f(e){return!(e&&"function"==typeof e&&e.apply&&!e[o])}},3606:(e,t,r)=>{"use strict";r.d(t,{i:()=>s});var n=r(9908);s.on=o;var i=s.handlers={};function s(e,t,r,s){o(s||n.d,i,e,t,r)}function o(e,t,r,i,s){s||(s="feature"),e||(e=n.d);var o=t[s]=t[s]||{};(o[r]=o[r]||[]).push([e,i])}},3738:(e,t,r)=>{"use strict";r.d(t,{He:()=>i,Kp:()=>a,Lc:()=>d,Rz:()=>u,TZ:()=>n,bD:()=>s,d3:()=>o,jx:()=>l,sl:()=>f,uP:()=>c});const n=r(860).K7.sessionTrace,i="bstResource",s="resource",o="-start",a="-end",c="fn"+o,d="fn"+a,u="pushState",l=1e3,f=3e4},3785:(e,t,r)=>{"use strict";r.d(t,{R:()=>c,b:()=>d});var n=r(9908),i=r(1863),s=r(860),o=r(3969),a=r(993);function c(e,t,r={},c=a.p_.INFO,d=!0,u,l=(0,i.t)()){(0,n.p)(o.xV,["API/logging/".concat(c.toLowerCase(),"/called")],void 0,s.K7.metrics,e),(0,n.p)(a.ET,[l,t,r,c,d,u],void 0,s.K7.logging,e)}function d(e){return"string"==typeof e&&Object.values(a.p_).some(t=>t===e.toUpperCase().trim())}},3878:(e,t,r)=>{"use strict";function n(e,t){return{capture:e,passive:!1,signal:t}}function i(e,t,r=!1,i){window.addEventListener(e,t,n(r,i))}function s(e,t,r=!1,i){document.addEventListener(e,t,n(r,i))}r.d(t,{DD:()=>s,jT:()=>n,sp:()=>i})},3962:(e,t,r)=>{"use strict";r.d(t,{AM:()=>o,O2:()=>l,OV:()=>s,Qu:()=>f,TZ:()=>c,ih:()=>h,pP:()=>a,t1:()=>u,tC:()=>i,wD:()=>d});var n=r(860);const i=["click","keydown","submit"],s="popstate",o="api",a="initialPageLoad",c=n.K7.softNav,d=5e3,u=500,l={INITIAL_PAGE_LOAD:"",ROUTE_CHANGE:1,UNSPECIFIED:2},f={INTERACTION:1,AJAX:2,CUSTOM_END:3,CUSTOM_TRACER:4},h={IP:"in progress",PF:"pending finish",FIN:"finished",CAN:"cancelled"}},3969:(e,t,r)=>{"use strict";r.d(t,{TZ:()=>n,XG:()=>a,rs:()=>i,xV:()=>o,z_:()=>s});const n=r(860).K7.metrics,i="sm",s="cm",o="storeSupportabilityMetrics",a="storeEventMetrics"},4234:(e,t,r)=>{"use strict";r.d(t,{W:()=>i});var n=r(1687);class i{constructor(e,t){this.agentRef=e,this.ee=e?.ee,this.featureName=t,this.blocked=!1}deregisterDrain(){(0,n.x3)(this.agentRef,this.featureName)}}},4261:(e,t,r)=>{"use strict";r.d(t,{$9:()=>u,BL:()=>c,CH:()=>p,Dl:()=>R,Fw:()=>w,PA:()=>v,Pl:()=>n,Pv:()=>x,Tb:()=>f,U2:()=>o,V1:()=>A,Wb:()=>T,bt:()=>b,cD:()=>y,d3:()=>E,dT:()=>d,eY:()=>g,fF:()=>h,hG:()=>s,hw:()=>i,k6:()=>a,nb:()=>m,o5:()=>l});const n="api-",i=n+"ixn-",s="addPageAction",o="addToTrace",a="addRelease",c="finished",d="interaction",u="log",l="noticeError",f="pauseReplay",h="recordCustomEvent",p="recordReplay",g="register",m="setApplicationVersion",v="setCurrentRouteName",y="setCustomAttribute",b="setErrorHandler",w="setPageViewName",R="setUserId",E="start",T="wrapLogger",A="measure",x="consent"},5205:(e,t,r)=>{"use strict";r.d(t,{j:()=>x});var n=r(384),i=r(1741);var s=r(2555),o=r(3333);const a=e=>{if(!e||"string"!=typeof e)return!1;try{document.createDocumentFragment().querySelector(e)}catch{return!1}return!0};var c=r(2614),d=r(944),u=r(8122);const l="[data-nr-mask]",f=e=>(0,u.a)(e,(()=>{const e={feature_flags:[],experimental:{register:!1,resources:!1},mask_selector:"*",block_selector:"[data-nr-block]",mask_input_options:{color:!1,date:!1,"datetime-local":!1,email:!1,month:!1,number:!1,range:!1,search:!1,tel:!1,text:!1,time:!1,url:!1,week:!1,textarea:!1,select:!1,password:!0}};return{ajax:{deny_list:void 0,block_internal:!0,enabled:!0,autoStart:!0},api:{register:{get enabled(){return e.feature_flags.includes(o.$v.REGISTER)||e.experimental.register},set enabled(t){e.experimental.register=t},duplicate_data_to_container:!1}},browser_consent_mode:{enabled:!1},distributed_tracing:{enabled:void 0,exclude_newrelic_header:void 0,cors_use_newrelic_header:void 0,cors_use_tracecontext_headers:void 0,allowed_origins:void 0},get feature_flags(){return e.feature_flags},set feature_flags(t){e.feature_flags=t},generic_events:{enabled:!0,autoStart:!0},harvest:{interval:30},jserrors:{enabled:!0,autoStart:!0},logging:{enabled:!0,autoStart:!0},metrics:{enabled:!0,autoStart:!0},obfuscate:void 0,page_action:{enabled:!0},page_view_event:{enabled:!0,autoStart:!0},page_view_timing:{enabled:!0,autoStart:!0},performance:{capture_marks:!1,capture_measures:!1,capture_detail:!0,resources:{get enabled(){return e.feature_flags.includes(o.$v.RESOURCES)||e.experimental.resources},set enabled(t){e.experimental.resources=t},asset_types:[],first_party_domains:[],ignore_newrelic:!0}},privacy:{cookies_enabled:!0},proxy:{assets:void 0,beacon:void 0},session:{expiresMs:c.wk,inactiveMs:c.BB},session_replay:{autoStart:!0,enabled:!1,preload:!1,sampling_rate:10,error_sampling_rate:100,collect_fonts:!1,inline_images:!1,fix_stylesheets:!0,mask_all_inputs:!0,get mask_text_selector(){return e.mask_selector},set mask_text_selector(t){a(t)?e.mask_selector="".concat(t,",").concat(l):""===t||null===t?e.mask_selector=l:(0,d.R)(5,t)},get block_class(){return"nr-block"},get ignore_class(){return"nr-ignore"},get mask_text_class(){return"nr-mask"},get block_selector(){return e.block_selector},set block_selector(t){a(t)?e.block_selector+=",".concat(t):""!==t&&(0,d.R)(6,t)},get mask_input_options(){return e.mask_input_options},set mask_input_options(t){t&&"object"==typeof t?e.mask_input_options={...t,password:!0}:(0,d.R)(7,t)}},session_trace:{enabled:!0,autoStart:!0},soft_navigations:{enabled:!0,autoStart:!0},ssl:void 0,user_actions:{enabled:!0,elementAttributes:["id","className","tagName","type"]}}})());var h=r(6154),p=r(9324);let g=0;const m={buildEnv:p.F3,distMethod:p.Xs,version:p.xv,originTime:h.WN},v={consented:!1},y={activatedFeatures:void 0,appMetadata:{},configured:!1,get consented(){return this.session?.state?.consent||v.consented},set consented(e){v.consented=e},customTransaction:void 0,denyList:[],disabled:!1,drainRegistry:new Map,harvester:void 0,isolatedBacklog:!1,isRecording:!1,loaderType:void 0,maxBytes:3e4,obfuscator:void 0,onerror:void 0,ptid:void 0,releaseIds:{},session:void 0,timeKeeper:void 0,registeredEntities:[],jsAttributesMetadata:{bytes:0},get harvestCount(){return++g}},b=e=>{const t=(0,u.a)(e,y),r=Object.keys(m).reduce((e,t)=>(e[t]={value:m[t],writable:!1,configurable:!0,enumerable:!0},e),{});return Object.defineProperties(t,r)},w=e=>{const t=e.startsWith("http");e+="/",r.p=t?e:"https://"+e};var R=r(7836),E=r(3241);const T={accountID:void 0,trustKey:void 0,agentID:void 0,licenseKey:void 0,applicationID:void 0,xpid:void 0},A=e=>(0,u.a)(e,T);function x(e,t={},r,o){let{init:a,info:c,loader_config:d,runtime:u={},exposed:l=!0}=t;if(!c){const e=(0,n.pV)();a=e.init,c=e.info,d=e.loader_config}e.init=f(a||{}),e.loader_config=A(d||{}),c.jsAttributes??={},h.bv&&(c.jsAttributes.isWorker=!0),e.info=(0,s.D)(c);const p=e.init;e.runtime??=b(u),p.proxy.assets&&w(p.proxy.assets),e.runtime.configured||(Object.defineProperty(e,"beacons",{get:()=>[e.info.beacon,e.info.errorBeacon,e.init.proxy.assets,e.init.proxy.beacon].filter(Boolean)}),Object.defineProperty(e.runtime,"denyList",{get:()=>[...e.init.ajax.deny_list||[],...e.init.ajax.block_internal?e.beacons:[]]}),e.runtime.ptid=e.agentIdentifier,function(e){const t=(0,n.pV)();Object.getOwnPropertyNames(i.W.prototype).forEach(r=>{const n=i.W.prototype[r];if("function"!=typeof n||"constructor"===n)return;let s=t[r];e[r]&&!1!==e.exposed&&"micro-agent"!==e.runtime?.loaderType&&(t[r]=(...t)=>{const n=e[r](...t);return s?s(...t):n})})}(e),e.runtime.loaderType=r,e.ee=R.ee.get(e.agentIdentifier),e.exposed=l,(0,E.W)({drained:!!e.runtime.activatedFeatures,type:"lifecycle",name:"initialize",feature:void 0,data:e.config}),e.runtime.configured=!0)}},5270:(e,t,r)=>{"use strict";r.d(t,{Aw:()=>o,SR:()=>s,rF:()=>a});var n=r(384),i=r(7767);function s(e){return!!(0,n.dV)().o.MO&&(0,i.V)(e)&&!0===e?.session_trace.enabled}function o(e){return!0===e?.session_replay.preload&&s(e)}function a(e,t){try{if("string"==typeof t?.type){if("password"===t.type.toLowerCase())return"*".repeat(e?.length||0);if(void 0!==t?.dataset?.nrUnmask||t?.classList?.contains("nr-unmask"))return e}}catch(e){}return"string"==typeof e?e.replace(/[\S]/g,"*"):"*".repeat(e?.length||0)}},5289:(e,t,r)=>{"use strict";r.d(t,{GG:()=>o,Qr:()=>c,sB:()=>a});var n=r(3878),i=r(6389);function s(){return"undefined"==typeof document||"complete"===document.readyState}function o(e,t){if(s())return e();const r=(0,i.J)(e),o=setInterval(()=>{s()&&(clearInterval(o),r())},500);(0,n.sp)("load",r,t)}function a(e){if(s())return e();(0,n.DD)("DOMContentLoaded",e)}function c(e){if(s())return e();(0,n.sp)("popstate",e)}},5607:(e,t,r)=>{"use strict";r.d(t,{W:()=>n});const n=(0,r(9566).bz)()},5732:(e,t,r)=>{"use strict";r.d(t,{$5:()=>u,B5:()=>d,Ms:()=>s,Ux:()=>a,YA:()=>c,fQ:()=>i,yx:()=>o});var n=r(7508);const i={MFE:"MFE",BA:"BA"};function s(e,t){if(!e||!t?.init.api.register.enabled)return[];const r=t.runtime.registeredEntities;return r?.filter(t=>String(t.metadata.target.id)===String(e)).map(e=>e.metadata.target)||[]}function o(e,t){if(!e||!t?.init.api.register.enabled)return[];const r=t.runtime.registeredEntities;return r?.filter(t=>t.metadata.timings?.asset?.endsWith(e)).map(e=>e.metadata.target)||[]}function a(e,t){if(!l(t))return{};const r=t.agentRef.runtime.appMetadata.agents[0].entityGuid;return e?e.attributes:{"entity.guid":r,appId:t.agentRef.info.applicationID}}function c(e,t){return d(e,t)?{"child.id":e.id,"child.type":e.type,...a(void 0,t)}:{}}function d(e,t){return!!e&&!!l(t)&&t.agentRef.init.api.register.duplicate_data_to_container}function u(e){if(!e?.init.api.register.enabled)return[void 0];const t=[];try{var r=(0,n.AZ)((0,n.QL)());let i=r.length-1;for(;r[i];)t.push(...o(r[i--],e))}catch(e){}return t.length||t.push(void 0),t}function l(e){return 2===e?.harvestEndpointVersion}},6154:(e,t,r)=>{"use strict";r.d(t,{OF:()=>d,RI:()=>i,WN:()=>f,bv:()=>s,gm:()=>o,lR:()=>l,m:()=>c,mw:()=>a,sb:()=>u,zk:()=>h});var n=r(1863);const i="undefined"!=typeof window&&!!window.document,s="undefined"!=typeof WorkerGlobalScope&&("undefined"!=typeof self&&self instanceof WorkerGlobalScope&&self.navigator instanceof WorkerNavigator||"undefined"!=typeof globalThis&&globalThis instanceof WorkerGlobalScope&&globalThis.navigator instanceof WorkerNavigator),o=i?window:"undefined"!=typeof WorkerGlobalScope&&("undefined"!=typeof self&&self instanceof WorkerGlobalScope&&self||"undefined"!=typeof globalThis&&globalThis instanceof WorkerGlobalScope&&globalThis),a=Boolean("hidden"===o?.document?.visibilityState),c=""+o?.location,d=/iPad|iPhone|iPod/.test(o.navigator?.userAgent),u=d&&"undefined"==typeof SharedWorker,l=(()=>{const e=o.navigator?.userAgent?.match(/Firefox[/\s](\d+\.\d+)/);return Array.isArray(e)&&e.length>=2?+e[1]:0})(),f=Date.now()-(0,n.t)(),h=()=>{const e=o?.performance?.getEntriesByType?.("navigation")?.[0];if(e&&e.responseStart>0&&e.responseStart<o.performance.now())return e}},6344:(e,t,r)=>{"use strict";r.d(t,{BB:()=>u,Qb:()=>l,TZ:()=>i,Ug:()=>o,Vh:()=>s,_s:()=>a,bc:()=>d,yP:()=>c});var n=r(2614);const i=r(860).K7.sessionReplay,s="errorDuringReplay",o=.12,a={DomContentLoaded:0,Load:1,FullSnapshot:2,IncrementalSnapshot:3,Meta:4,Custom:5},c={[n.g.ERROR]:15e3,[n.g.FULL]:3e5,[n.g.OFF]:0},d={RESET:{message:"Session was reset",sm:"Reset"},IMPORT:{message:"Recorder failed to import",sm:"Import"},TOO_MANY:{message:"429: Too Many Requests",sm:"Too-Many"},TOO_BIG:{message:"Payload was too large",sm:"Too-Big"},CROSS_TAB:{message:"Session Entity was set to OFF on another tab",sm:"Cross-Tab"},ENTITLEMENTS:{message:"Session Replay is not allowed and will not be started",sm:"Entitlement"}},u=5e3,l={API:"api",RESUME:"resume",SWITCH_TO_FULL:"switchToFull",INITIALIZE:"initialize",PRELOAD:"preload"}},6389:(e,t,r)=>{"use strict";function n(e,t=500,r={}){const n=r?.leading||!1;let i;return(...r)=>{n&&void 0===i&&(e.apply(this,r),i=setTimeout(()=>{i=clearTimeout(i)},t)),n||(clearTimeout(i),i=setTimeout(()=>{e.apply(this,r)},t))}}function i(e){let t=!1;return(...r)=>{t||(t=!0,e.apply(this,r))}}r.d(t,{J:()=>i,s:()=>n})},6630:(e,t,r)=>{"use strict";r.d(t,{T:()=>n});const n=r(860).K7.pageViewEvent},6774:(e,t,r)=>{"use strict";r.d(t,{T:()=>n});const n=r(860).K7.jserrors},7295:(e,t,r)=>{"use strict";r.d(t,{Xv:()=>o,gX:()=>i,iW:()=>s});var n=[];function i(e){if(!e||s(e))return!1;if(0===n.length)return!0;if("*"===n[0].hostname)return!1;for(var t=0;t<n.length;t++){var r=n[t];if(r.hostname.test(e.hostname)&&r.pathname.test(e.pathname))return!1}return!0}function s(e){return void 0===e.hostname}function o(e){if(n=[],e&&e.length)for(var t=0;t<e.length;t++){let r=e[t];if(!r)continue;if("*"===r)return void(n=[{hostname:"*"}]);0===r.indexOf("http://")?r=r.substring(7):0===r.indexOf("https://")&&(r=r.substring(8));const i=r.indexOf("/");let s,o;i>0?(s=r.substring(0,i),o=r.substring(i)):(s=r,o="*");let[c]=s.split(":");n.push({hostname:a(c),pathname:a(o,!0)})}}function a(e,t=!1){const r=e.replace(/[.+?^${}()|[\]\\]/g,e=>"\\"+e).replace(/\*/g,".*?");return new RegExp((t?"^":"")+r+"$")}},7485:(e,t,r)=>{"use strict";r.d(t,{D:()=>i});var n=r(6154);function i(e){if(0===(e||"").indexOf("data:"))return{protocol:"data"};try{const t=new URL(e,location.href),r={port:t.port,hostname:t.hostname,pathname:t.pathname,search:t.search,protocol:t.protocol.slice(0,t.protocol.indexOf(":")),sameOrigin:t.protocol===n.gm?.location?.protocol&&t.host===n.gm?.location?.host};return r.port&&""!==r.port||("http:"===t.protocol&&(r.port="80"),"https:"===t.protocol&&(r.port="443")),r.pathname&&""!==r.pathname?r.pathname.startsWith("/")||(r.pathname="/".concat(r.pathname)):r.pathname="/",r}catch(e){return{}}}},7508:(e,t,r)=>{"use strict";r.d(t,{AZ:()=>g,Qr:()=>b,QL:()=>m});var n=r(6154),i=r(1863),s=r(9119),o=r(7866);class a{dom=new c;performance=new c;constructor(e){this.url=e}get script(){const e=Math.max(this.dom.start,this.performance.end);return{start:e,end:Math.max(this.dom.end,this.performance.end,e)}}}class c{start=0;end=0;value=void 0}let d;try{d=g(m())[0]}catch(e){d=g(e)[0]}const u=e=>"script"===e.initiatorType||["link","fetch"].includes(e.initiatorType)&&e.name.endsWith(".js"),l=new Map;let f=[];function h(e){return l.get(e)}function p(e){const t=h(e);if(t)return t;const r=new a(e);if(l.set(e,r),l.size>1e3){const e=l.keys().next().value;l.delete(e)}return r}if(n.gm.MutationObserver&&n.gm.document){new MutationObserver(e=>{e.forEach(e=>{e.addedNodes.forEach(e=>{if("SCRIPT"===e.nodeName&&e.src){const t=p((0,s.L)(e.src));t.dom.start=(0,i.t)(),t.dom.value=e;const r=()=>{t.dom.end=(0,i.t)()};["load","error"].forEach(t=>e.addEventListener(t,r,{once:!0}))}})})}).observe(n.gm.document,{childList:!0,subtree:!0})}if(n.gm.PerformanceObserver?.supportedEntryTypes.includes("resource")){new PerformanceObserver(e=>{e.getEntries().filter(u).forEach(e=>{const t=p((0,s.L)(e.name));t.performance.start=Math.floor(e.startTime),t.performance.end=Math.floor(e.responseEnd),t.performance.value=e;const r=[];f.forEach(({test:t,addedAt:n},s)=>{(t(e)||(0,i.t)()-n>1e4)&&r.push(s)}),f=f.filter((e,t)=>!r.includes(t))})}).observe({type:"resource",buffered:!0})}function g(e){if(!e||"string"!=typeof e)return[];const t=new Set,r=e.split("\n");for(const e of r){const r=e.match(o.cn)||e.match(o.hB)||e.match(o.fL);if(r&&r[2])t.add((0,s.L)(r[2]));else{const r=e.match(/\(([^)]+\.js):\d+:\d+\)/)||e.match(/^\s+at\s+([^\s(]+\.js):\d+:\d+/);r&&r[1]&&t.add((0,s.L)(r[1]))}}return[...t]}function m(){let e;try{const t=Error.stackTraceLimit;Error.stackTraceLimit=50,e=(new Error).stack,Error.stackTraceLimit=t}catch(t){e=(new Error).stack}return e}function v(e,t){return(0,s.L)(e.name)===t}function y(e,t){e.fetchStart=Math.floor(t.startTime),e.fetchEnd=Math.floor(t.responseEnd),e.asset=t.name,e.type=t.initiatorType}function b(){const e={registeredAt:(0,i.t)(),reportedAt:void 0,fetchStart:0,fetchEnd:0,scriptStart:0,scriptEnd:0,asset:void 0,type:"unknown"},t=m();if(!t)return e;const r=n.gm.performance?.getEntriesByType("navigation")?.[0]?.name||"";try{const o=g(t),a=(o.length>1?o.filter(e=>d!==e):o)[0];if(!a)return e;if(r.includes(a))return e.asset=(0,s.L)(r),e.type="inline",e;e.correlation=h(a);const c=e.correlation?.performance.value||performance.getEntriesByType("resource").find(e=>v(e,a));c?y(e,c):function(e){if(!e||!n.gm.document)return!1;try{const t=n.gm.document.querySelectorAll('link[rel="preload"][as="script"]');for(const r of t)if((0,s.L)(r.href)===e)return!0}catch(e){}return!1}(a)&&(e.asset=a,e.type="preload",f.push({addedAt:(0,i.t)(),test:t=>!!v(t,a)&&(y(e,t),!0)})),Object.defineProperty(e,"scriptStart",{get:()=>e.correlation?.script.start||e.fetchEnd}),Object.defineProperty(e,"scriptEnd",{get:()=>e.correlation?.script.end||e.registeredAt})}catch(e){}return e}},7699:(e,t,r)=>{"use strict";r.d(t,{It:()=>s,KC:()=>a,No:()=>i,qh:()=>o});var n=r(860);const i=16e3,s=1e6,o="SESSION_ERROR",a={[n.K7.logging]:!0,[n.K7.genericEvents]:!0,[n.K7.jserrors]:!0,[n.K7.ajax]:!0}},7767:(e,t,r)=>{"use strict";r.d(t,{V:()=>i});var n=r(6154);const i=e=>n.RI&&!0===e?.privacy.cookies_enabled},7836:(e,t,r)=>{"use strict";r.d(t,{P:()=>a,ee:()=>c});var n=r(384),i=r(8990),s=r(2646),o=r(5607);const a="nr@context:".concat(o.W),c=function e(t,r){var n={},o={},u={},l=!1;try{l=16===r.length&&d.initializedAgents?.[r]?.runtime.isolatedBacklog}catch(e){}var f={on:p,addEventListener:p,removeEventListener:function(e,t){var r=n[e];if(!r)return;for(var i=0;i<r.length;i++)r[i]===t&&r.splice(i,1)},emit:function(e,r,n,i,s){!1!==s&&(s=!0);if(c.aborted&&!i)return;t&&s&&t.emit(e,r,n);var a=h(n);g(e).forEach(e=>{e.apply(a,r)});var d=v()[o[e]];d&&d.push([f,e,r,a]);return a},get:m,listeners:g,context:h,buffer:function(e,t){const r=v();if(t=t||"feature",f.aborted)return;Object.entries(e||{}).forEach(([e,n])=>{o[n]=t,t in r||(r[t]=[])})},abort:function(){f._aborted=!0,Object.keys(f.backlog).forEach(e=>{delete f.backlog[e]})},isBuffering:function(e){return!!v()[o[e]]},debugId:r,backlog:l?{}:t&&"object"==typeof t.backlog?t.backlog:{},isolatedBacklog:l};return Object.defineProperty(f,"aborted",{get:()=>{let e=f._aborted||!1;return e||(t&&(e=t.aborted),e)}}),f;function h(e){return e&&e instanceof s.y?e:e?(0,i.I)(e,a,()=>new s.y(a)):new s.y(a)}function p(e,t){n[e]=g(e).concat(t)}function g(e){return n[e]||[]}function m(t){return u[t]=u[t]||e(f,t)}function v(){return f.backlog}}(void 0,"globalEE"),d=(0,n.Zm)();d.ee||(d.ee=c)},7866:(e,t,r)=>{"use strict";r.d(t,{Nc:()=>s,cn:()=>a,fL:()=>i,h3:()=>n,hB:()=>o});const n=/function (.+?)\s*\(/,i=/^\s*at .+ \(eval at \S+ \((?:(?:file|http|https):[^)]+)?\)(?:, [^:]*:\d+:\d+)?\)$/i,s=/^\s*at Function code \(Function code:\d+:\d+\)\s*/i,o=/^\s*at (?:((?:\[object object\])?(?:[^(]*\([^)]*\))*[^()]*(?: \[as \S+\])?) )?\(?((?:file|http|https|chrome-extension):.*?)?:(\d+)(?::(\d+))?\)?\s*$/i,a=/^\s*(?:([^@]*)(?:\(.*?\))?@)?((?:file|http|https|chrome|safari-extension).*?):(\d+)(?::(\d+))?\s*$/i},8122:(e,t,r)=>{"use strict";r.d(t,{a:()=>i});var n=r(944);function i(e,t){try{if(!e||"object"!=typeof e)return(0,n.R)(3);if(!t||"object"!=typeof t)return(0,n.R)(4);const r=Object.create(Object.getPrototypeOf(t),Object.getOwnPropertyDescriptors(t)),s=0===Object.keys(r).length?e:r;for(let o in s)if(void 0!==e[o])try{if(null===e[o]){r[o]=null;continue}Array.isArray(e[o])&&Array.isArray(t[o])?r[o]=Array.from(new Set([...e[o],...t[o]])):e[o]instanceof Map||e[o]instanceof Set||e[o]instanceof Date||e[o]instanceof RegExp?r[o]=e[o]:"object"==typeof e[o]&&"object"==typeof t[o]?r[o]=i(e[o],t[o]):r[o]=e[o]}catch(e){r[o]||(0,n.R)(1,e)}return r}catch(e){(0,n.R)(2,e)}}},8139:(e,t,r)=>{"use strict";r.d(t,{u:()=>f});var n=r(7836),i=r(3434),s=r(8990),o=r(6154);const a={},c=o.gm.XMLHttpRequest,d="addEventListener",u="removeEventListener",l="nr@wrapped:".concat(n.P);function f(e){var t=function(e){return(e||n.ee).get("events")}(e);if(a[t.debugId]++)return t;a[t.debugId]=1;var r=(0,i.YM)(t,!0);function f(e){r.inPlace(e,[d,u],"-",p)}function p(e,t){return e[1]}return"getPrototypeOf"in Object&&(o.RI&&h(document,f),c&&h(c.prototype,f),h(o.gm,f)),t.on(d+"-start",function(e,t){var n=e[1];if(null!==n&&("function"==typeof n||"object"==typeof n)&&"newrelic"!==e[0]){var i=(0,s.I)(n,l,function(){var e={object:function(){if("function"!=typeof n.handleEvent)return;return n.handleEvent.apply(n,arguments)},function:n}[typeof n];return e?r(e,"fn-",null,e.name||"anonymous"):n});this.wrapped=e[1]=i}}),t.on(u+"-start",function(e){e[1]=this.wrapped||e[1]}),t}function h(e,t,...r){let n=e;for(;"object"==typeof n&&!Object.prototype.hasOwnProperty.call(n,d);)n=Object.getPrototypeOf(n);n&&t(n,...r)}},8362:(e,t,r)=>{"use strict";r.d(t,{d:()=>s});var n=r(9566),i=r(1741);class s extends i.W{agentIdentifier=(0,n.LA)(16)}},8374:(e,t,r)=>{r.nc=(()=>{try{return document?.currentScript?.nonce}catch(e){}return""})()},8990:(e,t,r)=>{"use strict";r.d(t,{I:()=>i});var n=Object.prototype.hasOwnProperty;function i(e,t,r){if(n.call(e,t))return e[t];var i=r();if(Object.defineProperty&&Object.keys)try{return Object.defineProperty(e,t,{value:i,writable:!0,enumerable:!1}),i}catch(e){}return e[t]=i,i}},9119:(e,t,r)=>{"use strict";r.d(t,{L:()=>s});var n=/([^?#]*)[^#]*(#[^?]*|$).*/,i=/([^?#]*)().*/;function s(e,t){return e?e.replace(t?n:i,"$1$2"):e}},9300:(e,t,r)=>{"use strict";r.d(t,{T:()=>n,f:()=>i});const n=r(860).K7.ajax,i="ajaxRequest.id"},9324:(e,t,r)=>{"use strict";r.d(t,{AJ:()=>o,F3:()=>i,Xs:()=>s,Yq:()=>a,xv:()=>n});const n="1.314.0",i="PROD",s="CDN",o="@newrelic/rrweb",a="1.1.0"},9566:(e,t,r)=>{"use strict";r.d(t,{LA:()=>a,ZF:()=>c,bz:()=>o,el:()=>d});var n=r(6154);const i="xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx";function s(e,t){return e?15&e[t]:16*Math.random()|0}function o(){const e=n.gm?.crypto||n.gm?.msCrypto;let t,r=0;return e&&e.getRandomValues&&(t=e.getRandomValues(new Uint8Array(30))),i.split("").map(e=>"x"===e?s(t,r++).toString(16):"y"===e?(3&s()|8).toString(16):e).join("")}function a(e){const t=n.gm?.crypto||n.gm?.msCrypto;let r,i=0;t&&t.getRandomValues&&(r=t.getRandomValues(new Uint8Array(e)));const o=[];for(var a=0;a<e;a++)o.push(s(r,i++).toString(16));return o.join("")}function c(){return a(16)}function d(){return a(32)}},9908:(e,t,r)=>{"use strict";r.d(t,{d:()=>n,p:()=>i});var n=r(7836).ee.get("handle");function i(e,t,r,i,s){s?(s.buffer([e],i),s.emit(e,t,r)):(n.buffer([e],i),n.emit(e,t,r))}}},n={};function i(e){var t=n[e];if(void 0!==t)return t.exports;var s=n[e]={exports:{}};return r[e](s,s.exports,i),s.exports}i.m=r,i.d=(e,t)=>{for(var r in t)i.o(t,r)&&!i.o(e,r)&&Object.defineProperty(e,r,{enumerable:!0,get:t[r]})},i.f={},i.e=e=>Promise.all(Object.keys(i.f).reduce((t,r)=>(i.f[r](e,t),t),[])),i.u=e=>({212:"nr-spa-compressor",249:"nr-spa-recorder",478:"nr-spa"}[e]+"-1.314.0.min.js"),i.o=(e,t)=>Object.prototype.hasOwnProperty.call(e,t),e={},t="NRBA-1.314.0.PROD:",i.l=(r,n,s,o)=>{if(e[r])e[r].push(n);else{var a,c;if(void 0!==s)for(var d=document.getElementsByTagName("script"),u=0;u<d.length;u++){var l=d[u];if(l.getAttribute("src")==r||l.getAttribute("data-webpack")==t+s){a=l;break}}if(!a){c=!0;var f={478:"sha512-mgj9qD5BdNJpEJCPmd1OfPzfnL/HRJ64OPYq/GZQxKc0756ytnQixcg8IA8iZWe/PRWN4DuGAnX82J2lscbiIw==",249:"sha512-fUJXIKjSt6UL9n1UZzNMzGGL3CxnBT/AgtoTat0/ONINbEB2PoiO+yK54DPYDwzpXDBddJHvs2BrYFj1GBpo2Q==",212:"sha512-vjoeBTgR0ONx6WcgBdfuPgcta1834zMPDhLL05C9c/pB+/xAuxL4BJMu9yeXYh+oIV7oIn6qTkDC7rBcZIDiHg=="};(a=document.createElement("script")).charset="utf-8",i.nc&&a.setAttribute("nonce",i.nc),a.setAttribute("data-webpack",t+s),a.src=r,0!==a.src.indexOf(window.location.origin+"/")&&(a.crossOrigin="anonymous"),f[o]&&(a.integrity=f[o])}e[r]=[n];var h=(t,n)=>{a.onerror=a.onload=null,clearTimeout(p);var i=e[r];if(delete e[r],a.parentNode&&a.parentNode.removeChild(a),i&&i.forEach(e=>e(n)),t)return t(n)},p=setTimeout(h.bind(null,void 0,{type:"timeout",target:a}),12e4);a.onerror=h.bind(null,a.onerror),a.onload=h.bind(null,a.onload),c&&document.head.appendChild(a)}},i.r=e=>{"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},i.p="https://js-agent.newrelic.com/",(()=>{var e={38:0,788:0};i.f.j=(t,r)=>{var n=i.o(e,t)?e[t]:void 0;if(0!==n)if(n)r.push(n[2]);else{var s=new Promise((r,i)=>n=e[t]=[r,i]);r.push(n[2]=s);var o=i.p+i.u(t),a=new Error;i.l(o,r=>{if(i.o(e,t)&&(0!==(n=e[t])&&(e[t]=void 0),n)){var s=r&&("load"===r.type?"missing":r.type),o=r&&r.target&&r.target.src;a.message="Loading chunk "+t+" failed: ("+s+": "+o+")",a.name="ChunkLoadError",a.type=s,a.request=o,n[1](a)}},"chunk-"+t,t)}};var t=(t,r)=>{var n,s,[o,a,c]=r,d=0;if(o.some(t=>0!==e[t])){for(n in a)i.o(a,n)&&(i.m[n]=a[n]);if(c)c(i)}for(t&&t(r);d<o.length;d++)s=o[d],i.o(e,s)&&e[s]&&e[s][0](),e[s]=0},r=self["webpackChunk:NRBA-1.314.0.PROD"]=self["webpackChunk:NRBA-1.314.0.PROD"]||[];r.forEach(t.bind(null,0)),r.push=t.bind(null,r.push.bind(r))})(),(()=>{"use strict";i(8374);var e=i(8362),t=i(860);const r=Object.values(t.K7);var n=i(5205);var s=i(9908),o=i(1863),a=i(4261),c=i(1738);var d=i(1687),u=i(4234),l=i(5289),f=i(6154),h=i(944),p=i(5270),g=i(7767),m=i(6389),v=i(7699);class y extends u.W{constructor(e,t){super(e,t),this.abortHandler=void 0,this.featAggregate=void 0,this.loadedSuccessfully=void 0,this.onAggregateImported=new Promise(e=>{this.loadedSuccessfully=e}),this.deferred=Promise.resolve(),!1===e.init[this.featureName].autoStart?this.deferred=new Promise((t,r)=>{this.ee.on("manual-start-all",(0,m.J)(()=>{(0,d.Ak)(e,this.featureName),t()}))}):(0,d.Ak)(e,t)}importAggregator(e,t,r={}){if(this.featAggregate)return;const n=async()=>{let n;await this.deferred;try{if((0,g.V)(e.init)){const{setupAgentSession:t}=await i.e(478).then(i.bind(i,8766));n=t(e)}}catch(e){(0,h.R)(20,e),this.ee.emit("internal-error",[e]),(0,s.p)(v.qh,[e],void 0,this.featureName,this.ee)}try{if(!this.#t(this.featureName,n,e.init))return(0,d.Ze)(this.agentRef,this.featureName),void this.loadedSuccessfully(!1);const{Aggregate:i}=await t();this.featAggregate=new i(e,r),e.runtime.harvester.initializedAggregates.push(this.featAggregate),this.loadedSuccessfully(!0)}catch(e){(0,h.R)(34,e),this.abortHandler?.(),(0,d.Ze)(this.agentRef,this.featureName,!0),this.loadedSuccessfully(!1),this.ee&&this.ee.abort()}};f.RI?(0,l.GG)(()=>n(),!0):n()}#t(e,r,n){if(this.blocked)return!1;switch(e){case t.K7.sessionReplay:return(0,p.SR)(n)&&!!r;case t.K7.sessionTrace:return!!r;default:return!0}}}var b=i(6630),w=i(2614),R=i(3241);class E extends y{static featureName=b.T;constructor(e){var t;super(e,b.T),this.setupInspectionEvents(),t=e,(0,c.Y)(a.Fw,function(e,r){"string"==typeof e&&("/"!==e.charAt(0)&&(e="/"+e),t.runtime.customTransaction=(r||"http://custom.transaction")+e,(0,s.p)(a.Pl+a.Fw,[(0,o.t)()],void 0,void 0,t.ee))},t),this.importAggregator(e,()=>i.e(478).then(i.bind(i,5839)))}setupInspectionEvents(){const e=(e,t)=>{e&&(0,R.W)({timeStamp:e.timeStamp,loaded:"complete"===e.target.readyState,type:"window",name:t,data:e.target.location+""})};(0,l.sB)(t=>{e(t,"DOMContentLoaded")}),(0,l.GG)(t=>{e(t,"load")}),(0,l.Qr)(t=>{e(t,"navigate")}),this.ee.on(w.tS.UPDATE,(e,t)=>{(0,R.W)({type:"lifecycle",name:"session",data:t})})}}var T=i(384);class A extends e.d{constructor(e){var t;(super(),f.gm)?(this.features={},(0,T.bQ)(this.agentIdentifier,this),this.desiredFeatures=new Set(e.features||[]),this.desiredFeatures.add(E),(0,n.j)(this,e,e.loaderType||"agent"),t=this,(0,c.Y)(a.cD,function(e,r,n=!1){if("string"==typeof e){if(["string","number","boolean"].includes(typeof r)||null===r)return(0,c.U)(t,e,r,a.cD,n);(0,h.R)(40,typeof r)}else(0,h.R)(39,typeof e)},t),function(e){(0,c.Y)(a.Dl,function(t,r=!1){if("string"!=typeof t&&null!==t)return void(0,h.R)(41,typeof t);const n=e.info.jsAttributes["enduser.id"];r&&null!=n&&n!==t?(0,s.p)(a.Pl+"setUserIdAndResetSession",[t],void 0,"session",e.ee):(0,c.U)(e,"enduser.id",t,a.Dl,!0)},e)}(this),function(e){(0,c.Y)(a.nb,function(t){if("string"==typeof t||null===t)return(0,c.U)(e,"application.version",t,a.nb,!1);(0,h.R)(42,typeof t)},e)}(this),function(e){(0,c.Y)(a.d3,function(){e.ee.emit("manual-start-all")},e)}(this),function(e){(0,c.Y)(a.Pv,function(t=!0){if("boolean"==typeof t){if((0,s.p)(a.Pl+a.Pv,[t],void 0,"session",e.ee),e.runtime.consented=t,t){const t=e.features.page_view_event;t.onAggregateImported.then(e=>{const r=t.featAggregate;e&&!r.sentRum&&r.sendRum()})}}else(0,h.R)(65,typeof t)},e)}(this),this.run()):(0,h.R)(21)}get config(){return{info:this.info,init:this.init,loader_config:this.loader_config,runtime:this.runtime}}get api(){return this}run(){try{const e=function(e){const t={};return r.forEach(r=>{t[r]=!!e[r]?.enabled}),t}(this.init),n=[...this.desiredFeatures];n.sort((e,r)=>t.P3[e.featureName]-t.P3[r.featureName]),n.forEach(r=>{if(!e[r.featureName]&&r.featureName!==t.K7.pageViewEvent)return;const n=function(e){switch(e){case t.K7.ajax:return[t.K7.jserrors];case t.K7.sessionTrace:return[t.K7.ajax,t.K7.pageViewEvent];case t.K7.sessionReplay:return[t.K7.sessionTrace];case t.K7.pageViewTiming:return[t.K7.pageViewEvent];default:return[]}}(r.featureName).filter(e=>!(e in this.features));n.length>0&&(0,h.R)(36,{targetFeature:r.featureName,missingDependencies:n}),this.features[r.featureName]=new r(this)})}catch(e){(0,h.R)(22,e);for(const e in this.features)this.features[e].abortHandler?.();const t=(0,T.Zm)();delete t.initializedAgents[this.agentIdentifier]?.features,delete this.sharedAggregator;return t.ee.get(this.agentIdentifier).abort(),!1}}}var x=i(2843),S=i(782);class _ extends y{static featureName=S.T;constructor(e){super(e,S.T),f.RI&&((0,x.u)(()=>(0,s.p)("docHidden",[(0,o.t)()],void 0,S.T,this.ee),!0),(0,x.G)(()=>(0,s.p)("winPagehide",[(0,o.t)()],void 0,S.T,this.ee)),this.importAggregator(e,()=>i.e(478).then(i.bind(i,9917))))}}var O=i(3969);class P extends y{static featureName=O.TZ;constructor(e){super(e,O.TZ),this.importAggregator(e,()=>i.e(478).then(i.bind(i,6555)))}}var k=i(6774),N=i(3878),j=i(3304);class D{constructor(e,t,r,n,i){this.name="UncaughtError",this.message="string"==typeof e?e:(0,j.A)(e),this.sourceURL=t,this.line=r,this.column=n,this.__newrelic=i}}function C(e){return M(e)?e:new D(void 0!==e?.message?e.message:e,e?.filename||e?.sourceURL,e?.lineno||e?.line,e?.colno||e?.col,e?.__newrelic,e?.cause)}function L(e){const t="Unhandled Promise Rejection: ";if(!e?.reason)return;if(M(e.reason)){try{e.reason.message.startsWith(t)||(e.reason.message=t+e.reason.message)}catch(e){}return C(e.reason)}const r=C(e.reason);return(r.message||"").startsWith(t)||(r.message=t+r.message),r}function I(e){if(e.error instanceof SyntaxError&&!/:\d+$/.test(e.error.stack?.trim())){const t=new D(e.message,e.filename,e.lineno,e.colno,e.error.__newrelic,e.cause);return t.name=SyntaxError.name,t}return M(e.error)?e.error:C(e)}function M(e){return e instanceof Error&&!!e.stack}function B(e,r,n,i,a=(0,o.t)()){"string"==typeof e&&(e=new Error(e)),(0,s.p)("err",[e,a,!1,r,n.runtime.isRecording,void 0,i],void 0,t.K7.jserrors,n.ee),(0,s.p)("uaErr",[],void 0,t.K7.genericEvents,n.ee)}var H=i(5732),K=i(993),W=i(3785);function F(e,{customAttributes:t={},level:r=K.p_.INFO}={},n,i,s=(0,o.t)()){(0,W.R)(n.ee,e,t,r,!1,i,s)}function U(e,r,n,i,c=(0,o.t)()){(0,s.p)(a.Pl+a.hG,[c,e,r,i],void 0,t.K7.genericEvents,n.ee)}function V(e,r,n,i,c=(0,o.t)()){const{start:d,end:u,customAttributes:l}=r||{},f={customAttributes:l||{}};if("object"!=typeof f.customAttributes||"string"!=typeof e||0===e.length)return void(0,h.R)(57);const p=(e,t)=>null==e?t:"number"==typeof e?e:e instanceof PerformanceMark?e.startTime:Number.NaN;if(f.start=p(d,0),f.end=p(u,c),Number.isNaN(f.start)||Number.isNaN(f.end))(0,h.R)(57);else{if(f.duration=f.end-f.start,!(f.duration<0))return(0,s.p)(a.Pl+a.V1,[f,e,i],void 0,t.K7.genericEvents,n.ee),f;(0,h.R)(58)}}function z(e,r={},n,i,c=(0,o.t)()){(0,s.p)(a.Pl+a.fF,[c,e,r,i],void 0,t.K7.genericEvents,n.ee)}var G=i(7508),Y=i(9566);const Z=["name","id","type"],q=new Map([[U,"addPageAction"],[F,"log"],[V,"measure"],[B,"noticeError"],[z,"recordCustomEvent"]]),X={experimental:(0,m.J)(()=>(0,h.R)(54,"newrelic.register")),disabled:(0,m.J)(()=>(0,h.R)(55)),invalidTarget:(0,m.J)(e=>(0,h.R)(48,e)),deregistered:(0,m.J)(()=>(0,h.R)(68))};function Q(e){(0,c.Y)(a.eY,function(t){return J(e,t)},e)}function J(e,r){X.experimental(),r||={},r.instance=(0,Y.LA)(8),r.type=H.fQ.MFE,r.licenseKey||=e.info.licenseKey,r.blocked=!1,("object"!=typeof r.tags||null===r.tags||Array.isArray(r.tags))&&(r.tags={}),r.parent??={get id(){return e.runtime.appMetadata.agents[0].entityGuid},type:H.fQ.BA};const n=(0,G.Qr)(),i={};Object.prototype.hasOwnProperty.call(r,"attributes")||Object.defineProperty(r,"attributes",{get:()=>({...i,"source.id":r.id,"source.name":r.name,"source.type":r.type,"parent.type":r.parent?.type||H.fQ.BA,"parent.id":r.parent?.id})}),Object.entries(r.tags).forEach(([e,t])=>{Z.includes(e)||(i["source.".concat(e)]=t)});let a=()=>{};const c=e.runtime.registeredEntities,d=e=>{r.blocked=!0,a=e};function u(e){return"string"==typeof e&&!!e.trim()&&e.trim().length<501}e.init.api.register.enabled||d(X.disabled),u(r.id)&&u(r.name)||d(()=>X.invalidTarget(r));const l={addPageAction:(t,n={})=>m(U,[t,{...i,...n},e],r),deregister:()=>{p(),d(X.deregistered)},log:(t,n={})=>m(F,[t,{...n,customAttributes:{...i,...n.customAttributes||{}}},e],r),measure:(t,n={})=>m(V,[t,{...n,customAttributes:{...i,...n.customAttributes||{}}},e],r),noticeError:(t,n={})=>m(B,[t,{...i,...n},e],r),recordCustomEvent:(t,n={})=>m(z,[t,{...i,...n},e],r),setApplicationVersion:e=>g("application.version",e),setCustomAttribute:(e,t)=>g(e,t),setUserId:e=>g("enduser.id",e),metadata:{get customAttributes(){return i},target:r,timings:n}},f=()=>(r.blocked&&a(),r.blocked);function p(){if(n.reportedAt)return;n.reportedAt=(0,o.t)();const e=n.fetchEnd-n.fetchStart,t=n.scriptEnd-n.scriptStart;l.recordCustomEvent("MicroFrontEndTiming",{assetUrl:n.asset,assetType:n.type,timeAlive:n.reportedAt-n.registeredAt,timeToBeRequested:n.fetchStart,timeToExecute:t,timeToFetch:e,timeToLoad:e+t,timeToRegister:n.registeredAt})}f()||(c.push(l),(0,x.G)(p));const g=(e,t)=>{f()||(i[e]=t)},m=(r,n,i)=>{if(f()&&r!==J)return;const a=(0,o.t)(),c=q.get(r)||"unknown";(0,s.p)(O.xV,["API/register/".concat(c,"/called")],void 0,t.K7.metrics,e.ee);try{return r(...n,i,a)}catch(e){(0,h.R)(50,e)}};return l}class ee extends y{static featureName=k.T;constructor(e){var t;super(e,k.T),t=e,(0,c.Y)(a.o5,(e,r)=>B(e,r,t),t),function(e){(0,c.Y)(a.bt,function(t){e.runtime.onerror=t},e)}(e),function(e){let t=0;(0,c.Y)(a.k6,function(e,r){++t>10||(this.runtime.releaseIds[e.slice(-200)]=(""+r).slice(-200))},e)}(e),Q(e);try{this.removeOnAbort=new AbortController}catch(e){}this.ee.on("internal-error",(t,r)=>{this.abortHandler&&(0,s.p)("ierr",[C(t),(0,o.t)(),!0,{},e.runtime.isRecording,r],void 0,this.featureName,this.ee)}),f.gm.addEventListener("unhandledrejection",t=>{this.abortHandler&&(0,s.p)("err",[L(t),(0,o.t)(),!1,{unhandledPromiseRejection:1},e.runtime.isRecording],void 0,this.featureName,this.ee)},(0,N.jT)(!1,this.removeOnAbort?.signal)),f.gm.addEventListener("error",t=>{this.abortHandler&&(0,s.p)("err",[I(t),(0,o.t)(),!1,{},e.runtime.isRecording],void 0,this.featureName,this.ee)},(0,N.jT)(!1,this.removeOnAbort?.signal)),this.abortHandler=this.#r,this.importAggregator(e,()=>i.e(478).then(i.bind(i,9377)))}#r(){this.removeOnAbort?.abort(),this.abortHandler=void 0}}var te=i(8990);let re=1;function ne(e){const t=typeof e;return!e||"object"!==t&&"function"!==t?-1:e===f.gm?0:(0,te.I)(e,"nr@id",function(){return re++})}function ie(e){if("string"==typeof e&&e.length)return e.length;if("object"==typeof e){if("undefined"!=typeof ArrayBuffer&&e instanceof ArrayBuffer&&e.byteLength)return e.byteLength;if("undefined"!=typeof Blob&&e instanceof Blob&&e.size)return e.size;if(!("undefined"!=typeof FormData&&e instanceof FormData))try{return(0,j.A)(e).length}catch(e){return}}}var se=i(8139),oe=i(7836),ae=i(3434);const ce={},de=["open","send"];function ue(e,t){var r=e||oe.ee;const n=function(e){return(e||oe.ee).get("xhr")}(r);if(void 0===f.gm.XMLHttpRequest)return n;if(ce[n.debugId]++)return n;ce[n.debugId]=1,(0,se.u)(r);var i=(0,ae.YM)(n),s=f.gm.XMLHttpRequest,o=f.gm.MutationObserver,a=f.gm.Promise,c=f.gm.setInterval,d="readystatechange",u=["onload","onerror","onabort","onloadstart","onloadend","onprogress","ontimeout"],l=[],p=f.gm.XMLHttpRequest=function(e){const r=new s(e),o=n.context(r);o.targets=(0,H.$5)(t);try{n.emit("new-xhr",[r],o),r.addEventListener(d,(a=o,function(){var e=this;e.readyState>3&&!a.resolved&&(a.resolved=!0,n.emit("xhr-resolved",[],e)),i.inPlace(e,u,"fn-",w)}),(0,N.jT)(!1))}catch(e){(0,h.R)(15,e);try{n.emit("internal-error",[e])}catch(e){}}var a;return r};function g(e,t){i.inPlace(t,["onreadystatechange"],"fn-",w)}if(function(e,t){for(var r in e)t[r]=e[r]}(s,p),p.prototype=s.prototype,i.inPlace(p.prototype,de,"-xhr-",w),n.on("send-xhr-start",function(e,t){g(e,t),function(e){l.push(e),o&&(m?m.then(b):c?c(b):(v=-v,y.data=v))}(t)}),n.on("open-xhr-start",g),o){var m=a&&a.resolve();if(!c&&!a){var v=1,y=document.createTextNode(v);new o(b).observe(y,{characterData:!0})}}else r.on("fn-end",function(e){e[0]&&e[0].type===d||b()});function b(){for(var e=0;e<l.length;e++)g(0,l[e]);l.length&&(l=[])}function w(e,t){return t}return n}var le="fetch-",fe=le+"body-",he=["arrayBuffer","blob","json","text","formData"],pe=f.gm.Request,ge=f.gm.Response,me="prototype";const ve={};function ye(e,t){const r=function(e){return(e||oe.ee).get("fetch")}(e);if(!(pe&&ge&&f.gm.fetch))return r;if(ve[r.debugId]++)return r;function n(e,n,i){var s=e[n];"function"==typeof s&&(e[n]=function(){var e=[...arguments];const n={},o=(0,H.$5)(t);var a;r.emit(i+"before-start",[e],n),n[oe.P]&&n[oe.P].dt&&(a=n[oe.P].dt);var c=s.apply(this,e);return r.emit(i+"start",[e,a],c),c.then(function(e){return r.emit(i+"end",[null,e,o],c),e},function(e){throw r.emit(i+"end",[e,void 0,o],c),e})})}return ve[r.debugId]=1,he.forEach(e=>{n(pe[me],e,fe),n(ge[me],e,fe)}),n(f.gm,"fetch",le),r.on(le+"end",function(e,t,n){var i=this;if(i.targets=n||[void 0],t){var s=t.headers.get("content-length");null!==s&&(i.rxSize=s),r.emit(le+"done",[null,t],i)}else r.emit(le+"done",[e],i)}),r}var be=i(7485);class we{constructor(e){this.agentRef=e}generateTracePayload(e){const t=this.agentRef.loader_config;if(!this.shouldGenerateTrace(e)||!t)return null;var r=(t.accountID||"").toString()||null,n=(t.agentID||"").toString()||null,i=(t.trustKey||"").toString()||null;if(!r||!n)return null;var s=(0,Y.ZF)(),o=(0,Y.el)(),a=Date.now(),c={spanId:s,traceId:o,timestamp:a};return(e.sameOrigin||this.isAllowedOrigin(e)&&this.useTraceContextHeadersForCors())&&(c.traceContextParentHeader=this.generateTraceContextParentHeader(s,o),c.traceContextStateHeader=this.generateTraceContextStateHeader(s,a,r,n,i)),(e.sameOrigin&&!this.excludeNewrelicHeader()||!e.sameOrigin&&this.isAllowedOrigin(e)&&this.useNewrelicHeaderForCors())&&(c.newrelicHeader=this.generateTraceHeader(s,o,a,r,n,i)),c}generateTraceContextParentHeader(e,t){return"00-"+t+"-"+e+"-01"}generateTraceContextStateHeader(e,t,r,n,i){return i+"@nr=0-1-"+r+"-"+n+"-"+e+"----"+t}generateTraceHeader(e,t,r,n,i,s){if(!("function"==typeof f.gm?.btoa))return null;var o={v:[0,1],d:{ty:"Browser",ac:n,ap:i,id:e,tr:t,ti:r}};return s&&n!==s&&(o.d.tk=s),btoa((0,j.A)(o))}shouldGenerateTrace(e){return this.agentRef.init?.distributed_tracing?.enabled&&this.isAllowedOrigin(e)}isAllowedOrigin(e){var t=!1;const r=this.agentRef.init?.distributed_tracing;if(e.sameOrigin)t=!0;else if(r?.allowed_origins instanceof Array)for(var n=0;n<r.allowed_origins.length;n++){var i=(0,be.D)(r.allowed_origins[n]);if(e.hostname===i.hostname&&e.protocol===i.protocol&&e.port===i.port){t=!0;break}}return t}excludeNewrelicHeader(){var e=this.agentRef.init?.distributed_tracing;return!!e&&!!e.exclude_newrelic_header}useNewrelicHeaderForCors(){var e=this.agentRef.init?.distributed_tracing;return!!e&&!1!==e.cors_use_newrelic_header}useTraceContextHeadersForCors(){var e=this.agentRef.init?.distributed_tracing;return!!e&&!!e.cors_use_tracecontext_headers}}var Re=i(9300),Ee=i(7295);function Te(e){return"string"==typeof e?e:e instanceof(0,T.dV)().o.REQ?e.url:f.gm?.URL&&e instanceof URL?e.href:void 0}var Ae=["load","error","abort","timeout"],xe=Ae.length,Se=(0,T.dV)().o.REQ,_e=(0,T.dV)().o.XHR;const Oe="X-NewRelic-App-Data";class Pe extends y{static featureName=Re.T;constructor(e){super(e,Re.T),this.dt=new we(e),this.handler=(e,t,r,n)=>(0,s.p)(e,t,r,n,this.ee);try{const e={xmlhttprequest:"xhr",fetch:"fetch",beacon:"beacon"};f.gm?.performance?.getEntriesByType("resource").forEach(r=>{if(r.initiatorType in e&&0!==r.responseStatus){const n={status:r.responseStatus},i={rxSize:r.transferSize,duration:Math.floor(r.duration),cbTime:0};ke(n,r.name),this.handler("xhr",[n,i,r.startTime,r.responseEnd,e[r.initiatorType]],void 0,t.K7.ajax)}})}catch(e){}ye(this.ee,e),ue(this.ee,e),function(e,r,n,i){function a(e){var t=this;t.totalCbs=0,t.called=0,t.cbTime=0,t.end=T,t.ended=!1,t.xhrGuids={},t.lastSize=null,t.loadCaptureCalled=!1,t.params=this.params||{},t.metrics=this.metrics||{},t.latestLongtaskEnd=0,e.addEventListener("load",function(r){x(t,e)},(0,N.jT)(!1)),f.lR||e.addEventListener("progress",function(e){t.lastSize=e.loaded},(0,N.jT)(!1))}function c(e){this.params={method:e[0]},ke(this,e[1]),this.metrics={}}function d(t,r){e.loader_config.xpid&&this.sameOrigin&&r.setRequestHeader("X-NewRelic-ID",e.loader_config.xpid);var n=i.generateTracePayload(this.parsedOrigin);if(n){var s=!1;n.newrelicHeader&&(r.setRequestHeader("newrelic",n.newrelicHeader),s=!0),n.traceContextParentHeader&&(r.setRequestHeader("traceparent",n.traceContextParentHeader),n.traceContextStateHeader&&r.setRequestHeader("tracestate",n.traceContextStateHeader),s=!0),s&&(this.dt=n)}}function u(e,t){var n=this.metrics,i=e[0],s=this;if(n&&i){var a=ie(i);a&&(n.txSize=a)}this.startTime=(0,o.t)(),this.body=i,this.listener=function(e){try{"abort"!==e.type||s.loadCaptureCalled||(s.params.aborted=!0),("load"!==e.type||s.called===s.totalCbs&&(s.onloadCalled||"function"!=typeof t.onload)&&"function"==typeof s.end)&&s.end(t)}catch(e){try{r.emit("internal-error",[e])}catch(e){}}};for(var c=0;c<xe;c++)t.addEventListener(Ae[c],this.listener,(0,N.jT)(!1))}function l(e,t,r){this.cbTime+=e,t?this.onloadCalled=!0:this.called+=1,this.called!==this.totalCbs||!this.onloadCalled&&"function"==typeof r.onload||"function"!=typeof this.end||this.end(r)}function h(e,t){var r=""+ne(e)+!!t;this.xhrGuids&&!this.xhrGuids[r]&&(this.xhrGuids[r]=!0,this.totalCbs+=1)}function p(e,t){var r=""+ne(e)+!!t;this.xhrGuids&&this.xhrGuids[r]&&(delete this.xhrGuids[r],this.totalCbs-=1)}function g(){this.endTime=(0,o.t)()}function m(e,t){t instanceof _e&&"load"===e[0]&&r.emit("xhr-load-added",[e[1],e[2]],t)}function v(e,t){t instanceof _e&&"load"===e[0]&&r.emit("xhr-load-removed",[e[1],e[2]],t)}function y(e,t,r){t instanceof _e&&("onload"===r&&(this.onload=!0),("load"===(e[0]&&e[0].type)||this.onload)&&(this.xhrCbStart=(0,o.t)()))}function b(e,t){this.xhrCbStart&&r.emit("xhr-cb-time",[(0,o.t)()-this.xhrCbStart,this.onload,t],t)}function w(e){var t,r=e[1]||{};if("string"==typeof e[0]?0===(t=e[0]).length&&f.RI&&(t=""+f.gm.location.href):e[0]&&e[0].url?t=e[0].url:f.gm?.URL&&e[0]&&e[0]instanceof URL?t=e[0].href:"function"==typeof e[0].toString&&(t=e[0].toString()),"string"==typeof t&&0!==t.length){t&&(this.parsedOrigin=(0,be.D)(t),this.sameOrigin=this.parsedOrigin.sameOrigin);var n=i.generateTracePayload(this.parsedOrigin);if(n&&(n.newrelicHeader||n.traceContextParentHeader))if(e[0]&&e[0].headers)a(e[0].headers,n)&&(this.dt=n);else{var s={};for(var o in r)s[o]=r[o];s.headers=new Headers(r.headers||{}),a(s.headers,n)&&(this.dt=n),e.length>1?e[1]=s:e.push(s)}}function a(e,t){var r=!1;return t.newrelicHeader&&(e.set("newrelic",t.newrelicHeader),r=!0),t.traceContextParentHeader&&(e.set("traceparent",t.traceContextParentHeader),t.traceContextStateHeader&&e.set("tracestate",t.traceContextStateHeader),r=!0),r}}function R(e,t){this.params={},this.metrics={},this.startTime=(0,o.t)(),this.dt=t;let[r,n={}]=e;ke(this,Te(r));const i=(""+(r&&r instanceof Se&&r.method||n.method||"GET")).toUpperCase();this.params.method=i,this.body=n.body,this.txSize=ie(n.body)||0}function E(e,t){if(this.endTime=(0,o.t)(),this.params||(this.params={}),(0,Ee.iW)(this.params))return;let r;this.params.status=t?t.status:0,"string"==typeof this.rxSize&&this.rxSize.length>0&&(r=+this.rxSize);const n={txSize:this.txSize,rxSize:r,duration:(0,o.t)()-this.startTime},i=[this.params,n,this.startTime,this.endTime,"fetch"];this.targets.forEach(e=>A(i,this,e))}function T(e){const t=this.params,r=this.metrics;if(this.ended)return;this.ended=!0;for(let t=0;t<xe;t++)e.removeEventListener(Ae[t],this.listener,!1);if(t.aborted)return;if((0,Ee.iW)(t))return;r.duration=(0,o.t)()-this.startTime,this.loadCaptureCalled||4!==e.readyState?null==t.status&&(t.status=0):x(this,e),r.cbTime=this.cbTime;const n=[t,r,this.startTime,this.endTime,"xhr"];this.targets.forEach(e=>A(n,this,e))}function A(e,r,i){n("xhr",[...e,i],r,t.K7.ajax)}function x(e,n){e.params.status=n.status;var i=function(e,t){var r=e.responseType;return"json"===r&&null!==t?t:"arraybuffer"===r||"blob"===r||"json"===r?ie(e.response):"text"===r||""===r||void 0===r?ie(e.responseText):void 0}(n,e.lastSize);if(i&&(e.metrics.rxSize=i),e.sameOrigin&&n.getAllResponseHeaders().indexOf(Oe)>=0){var o=n.getResponseHeader(Oe);o&&((0,s.p)(O.rs,["Ajax/CrossApplicationTracing/Header/Seen"],void 0,t.K7.metrics,r),e.params.cat=o.split(", ").pop())}e.loadCaptureCalled=!0}r.on("new-xhr",a),r.on("open-xhr-start",c),r.on("open-xhr-end",d),r.on("send-xhr-start",u),r.on("xhr-cb-time",l),r.on("xhr-load-added",h),r.on("xhr-load-removed",p),r.on("xhr-resolved",g),r.on("addEventListener-end",m),r.on("removeEventListener-end",v),r.on("fn-end",b),r.on("fetch-before-start",w),r.on("fetch-start",R),r.on("fn-start",y),r.on("fetch-done",E)}(e,this.ee,this.handler,this.dt),this.importAggregator(e,()=>i.e(478).then(i.bind(i,3845)))}}function ke(e,t){var r=(0,be.D)(t),n=e.params||e;n.hostname=r.hostname,n.port=r.port,n.protocol=r.protocol,n.host=r.hostname+":"+r.port,n.pathname=r.pathname,e.parsedOrigin=r,e.sameOrigin=r.sameOrigin}const Ne={},je=["pushState","replaceState"];function De(e){const t=function(e){return(e||oe.ee).get("history")}(e);return!f.RI||Ne[t.debugId]++||(Ne[t.debugId]=1,(0,ae.YM)(t).inPlace(window.history,je,"-")),t}var Ce=i(3738);function Le(e){(0,c.Y)(a.BL,function(r=Date.now()){const n=r-f.WN;n<0&&(0,h.R)(62,r),(0,s.p)(O.XG,[a.BL,{time:n}],void 0,t.K7.metrics,e.ee),e.addToTrace({name:a.BL,start:r,origin:"nr"}),(0,s.p)(a.Pl+a.hG,[n,a.BL],void 0,t.K7.genericEvents,e.ee)},e)}const{He:Ie,bD:Me,d3:Be,Kp:He,TZ:Ke,Lc:We,uP:Fe,Rz:Ue}=Ce;class Ve extends y{static featureName=Ke;constructor(e){var r;super(e,Ke),r=e,(0,c.Y)(a.U2,function(e){if(!(e&&"object"==typeof e&&e.name&&e.start))return;const n={n:e.name,s:e.start-f.WN,e:(e.end||e.start)-f.WN,o:e.origin||"",t:"api"};n.s<0||n.e<0||n.e<n.s?(0,h.R)(61,{start:n.s,end:n.e}):(0,s.p)("bstApi",[n],void 0,t.K7.sessionTrace,r.ee)},r),Le(e);if(!(0,g.V)(e.init))return void this.deregisterDrain();const n=this.ee;let d;De(n),this.eventsEE=(0,se.u)(n),this.eventsEE.on(Fe,function(e,t){this.bstStart=(0,o.t)()}),this.eventsEE.on(We,function(e,r){(0,s.p)("bst",[e[0],r,this.bstStart,(0,o.t)()],void 0,t.K7.sessionTrace,n)}),n.on(Ue+Be,function(e){this.time=(0,o.t)(),this.startPath=location.pathname+location.hash}),n.on(Ue+He,function(e){(0,s.p)("bstHist",[location.pathname+location.hash,this.startPath,this.time],void 0,t.K7.sessionTrace,n)});try{d=new PerformanceObserver(e=>{const r=e.getEntries();(0,s.p)(Ie,[r],void 0,t.K7.sessionTrace,n)}),d.observe({type:Me,buffered:!0})}catch(e){}this.importAggregator(e,()=>i.e(478).then(i.bind(i,6974)),{resourceObserver:d})}}var ze=i(6344);class Ge extends y{static featureName=ze.TZ;#n;recorder;constructor(e){var r;let n;super(e,ze.TZ),r=e,(0,c.Y)(a.CH,function(){(0,s.p)(a.CH,[],void 0,t.K7.sessionReplay,r.ee)},r),function(e){(0,c.Y)(a.Tb,function(){(0,s.p)(a.Tb,[],void 0,t.K7.sessionReplay,e.ee)},e)}(e);try{n=JSON.parse(localStorage.getItem("".concat(w.H3,"_").concat(w.uh)))}catch(e){}(0,p.SR)(e.init)&&this.ee.on(a.CH,()=>this.#i()),this.#s(n)&&this.importRecorder().then(e=>{e.startRecording(ze.Qb.PRELOAD,n?.sessionReplayMode)}),this.importAggregator(this.agentRef,()=>i.e(478).then(i.bind(i,6167)),this),this.ee.on("err",e=>{this.blocked||this.agentRef.runtime.isRecording&&(this.errorNoticed=!0,(0,s.p)(ze.Vh,[e],void 0,this.featureName,this.ee))})}#s(e){return e&&(e.sessionReplayMode===w.g.FULL||e.sessionReplayMode===w.g.ERROR)||(0,p.Aw)(this.agentRef.init)}importRecorder(){return this.recorder?Promise.resolve(this.recorder):(this.#n??=Promise.all([i.e(478),i.e(249)]).then(i.bind(i,4866)).then(({Recorder:e})=>(this.recorder=new e(this),this.recorder)).catch(e=>{throw this.ee.emit("internal-error",[e]),this.blocked=!0,e}),this.#n)}#i(){this.blocked||(this.featAggregate?this.featAggregate.mode!==w.g.FULL&&this.featAggregate.initializeRecording(w.g.FULL,!0,ze.Qb.API):this.importRecorder().then(()=>{this.recorder.startRecording(ze.Qb.API,w.g.FULL)}))}}var Ye=i(3962);class Ze extends y{static featureName=Ye.TZ;constructor(e){if(super(e,Ye.TZ),function(e){const r=e.ee.get("tracer");function n(){}(0,c.Y)(a.dT,function(e){return(new n).get("object"==typeof e?e:{})},e);const i=n.prototype={createTracer:function(n,i){var a={},c=this,d="function"==typeof i;return(0,s.p)(O.xV,["API/createTracer/called"],void 0,t.K7.metrics,e.ee),function(){if(r.emit((d?"":"no-")+"fn-start",[(0,o.t)(),c,d],a),d)try{return i.apply(this,arguments)}catch(e){const t="string"==typeof e?new Error(e):e;throw r.emit("fn-err",[arguments,this,t],a),t}finally{r.emit("fn-end",[(0,o.t)()],a)}}}};["actionText","setName","setAttribute","save","ignore","onEnd","getContext","end","get"].forEach(r=>{c.Y.apply(this,[r,function(){return(0,s.p)(a.hw+r,[performance.now(),...arguments],this,t.K7.softNav,e.ee),this},e,i])}),(0,c.Y)(a.PA,function(){(0,s.p)(a.hw+"routeName",[performance.now(),...arguments],void 0,t.K7.softNav,e.ee)},e)}(e),!f.RI||!(0,T.dV)().o.MO)return;const r=De(this.ee);try{this.removeOnAbort=new AbortController}catch(e){}Ye.tC.forEach(e=>{(0,N.sp)(e,e=>{l(e)},!0,this.removeOnAbort?.signal)});const n=()=>(0,s.p)("newURL",[(0,o.t)(),""+window.location],void 0,this.featureName,this.ee);r.on("pushState-end",n),r.on("replaceState-end",n),(0,N.sp)(Ye.OV,e=>{l(e),(0,s.p)("newURL",[e.timeStamp,""+window.location],void 0,this.featureName,this.ee)},!0,this.removeOnAbort?.signal);let d=!1;const u=new((0,T.dV)().o.MO)((e,t)=>{d||(d=!0,requestAnimationFrame(()=>{(0,s.p)("newDom",[(0,o.t)()],void 0,this.featureName,this.ee),d=!1}))}),l=(0,m.s)(e=>{"loading"!==document.readyState&&((0,s.p)("newUIEvent",[e],void 0,this.featureName,this.ee),u.observe(document.body,{attributes:!0,childList:!0,subtree:!0,characterData:!0}))},100,{leading:!0});this.abortHandler=function(){this.removeOnAbort?.abort(),u.disconnect(),this.abortHandler=void 0},this.importAggregator(e,()=>i.e(478).then(i.bind(i,4393)),{domObserver:u})}}var qe=i(3333),Xe=i(9119);const $e={},Qe=new Set;function Je(e){return"string"==typeof e?{type:"string",size:(new TextEncoder).encode(e).length}:e instanceof ArrayBuffer?{type:"ArrayBuffer",size:e.byteLength}:e instanceof Blob?{type:"Blob",size:e.size}:e instanceof DataView?{type:"DataView",size:e.byteLength}:ArrayBuffer.isView(e)?{type:"TypedArray",size:e.byteLength}:{type:"unknown",size:0}}class et{constructor(e,t){this.timestamp=(0,o.t)(),this.currentUrl=(0,Xe.L)(window.location.href),this.socketId=(0,Y.LA)(8),this.requestedUrl=(0,Xe.L)(e),this.requestedProtocols=Array.isArray(t)?t.join(","):t||"",this.openedAt=void 0,this.protocol=void 0,this.extensions=void 0,this.binaryType=void 0,this.messageOrigin=void 0,this.messageCount=0,this.messageBytes=0,this.messageBytesMin=0,this.messageBytesMax=0,this.messageTypes=void 0,this.sendCount=0,this.sendBytes=0,this.sendBytesMin=0,this.sendBytesMax=0,this.sendTypes=void 0,this.closedAt=void 0,this.closeCode=void 0,this.closeReason="unknown",this.closeWasClean=void 0,this.connectedDuration=0,this.hasErrors=void 0}}class tt extends y{static featureName=qe.TZ;constructor(e){super(e,qe.TZ);const r=e.init.feature_flags.includes("websockets"),n=!e.init.feature_flags.includes("no_spv"),d=[e.init.page_action.enabled,e.init.performance.capture_marks,e.init.performance.capture_measures,e.init.performance.resources.enabled,e.init.user_actions.enabled,r,n];var u;let l;if(u=e,(0,c.Y)(a.hG,(e,t)=>U(e,t,u),u),function(e){(0,c.Y)(a.fF,(t,r)=>z(t,r,e),e)}(e),Le(e),Q(e),function(e){(0,c.Y)(a.V1,(t,r)=>V(t,r,e),e)}(e),this.removeOnAbort=new AbortController,this.abortHandler=()=>{this.removeOnAbort.abort(),this.abortHandler=void 0},r){const h=function(e){if(!(0,T.dV)().o.WS)return e;const t=e.get("websockets");if($e[t.debugId]++)return t;$e[t.debugId]=1,(0,x.G)(()=>{const e=(0,o.t)();Qe.forEach(r=>{r.nrData.closedAt=e,r.nrData.closeCode=1001,r.nrData.closeReason="Page navigating away",r.nrData.closeWasClean=!1,r.nrData.openedAt&&(r.nrData.connectedDuration=e-r.nrData.openedAt),t.emit("ws",[r.nrData],r)})});class r extends WebSocket{static name="WebSocket";static toString(){return"function WebSocket() { [native code] }"}toString(){return"[object WebSocket]"}get[Symbol.toStringTag](){return r.name}#o(e){(e.__newrelic??={}).socketId=this.nrData.socketId,this.nrData.hasErrors??=!0}constructor(...e){super(...e),this.nrData=new et(e[0],e[1]),this.addEventListener("open",()=>{this.nrData.openedAt=(0,o.t)(),["protocol","extensions","binaryType"].forEach(e=>{this.nrData[e]=this[e]}),Qe.add(this)}),this.addEventListener("message",e=>{const{type:t,size:r}=Je(e.data);this.nrData.messageOrigin??=(0,Xe.L)(e.origin),this.nrData.messageCount++,this.nrData.messageBytes+=r,this.nrData.messageBytesMin=Math.min(this.nrData.messageBytesMin||1/0,r),this.nrData.messageBytesMax=Math.max(this.nrData.messageBytesMax,r),(this.nrData.messageTypes??"").includes(t)||(this.nrData.messageTypes=this.nrData.messageTypes?"".concat(this.nrData.messageTypes,",").concat(t):t)}),this.addEventListener("close",e=>{this.nrData.closedAt=(0,o.t)(),this.nrData.closeCode=e.code,e.reason&&(this.nrData.closeReason=e.reason),this.nrData.closeWasClean=e.wasClean,this.nrData.connectedDuration=this.nrData.closedAt-this.nrData.openedAt,Qe.delete(this),t.emit("ws",[this.nrData],this)})}addEventListener(e,t,...r){const n=this,i="function"==typeof t?function(...e){try{return t.apply(this,e)}catch(e){throw n.#o(e),e}}:t?.handleEvent?{handleEvent:function(...e){try{return t.handleEvent.apply(t,e)}catch(e){throw n.#o(e),e}}}:t;return super.addEventListener(e,i,...r)}send(e){if(this.readyState===WebSocket.OPEN){const{type:t,size:r}=Je(e);this.nrData.sendCount++,this.nrData.sendBytes+=r,this.nrData.sendBytesMin=Math.min(this.nrData.sendBytesMin||1/0,r),this.nrData.sendBytesMax=Math.max(this.nrData.sendBytesMax,r),(this.nrData.sendTypes??"").includes(t)||(this.nrData.sendTypes=this.nrData.sendTypes?"".concat(this.nrData.sendTypes,",").concat(t):t)}try{return super.send(e)}catch(e){throw this.#o(e),e}}close(...e){try{super.close(...e)}catch(e){throw this.#o(e),e}}}return f.gm.WebSocket=r,t}(this.ee);h.on("ws",e=>{(0,s.p)("ws-complete",[e],void 0,this.featureName,this.ee)})}if(n&&f.gm.addEventListener("securitypolicyviolation",e=>{(0,s.p)("spv",[e],void 0,t.K7.genericEvents,this.ee)},(0,N.jT)(!1,this.removeOnAbort.signal)),f.RI){if(ye(this.ee,e),ue(this.ee,e),l=De(this.ee),e.init.user_actions.enabled){function p(t){const r=(0,be.D)(t);return e.beacons.includes(r.hostname+":"+r.port)}function g(){l.emit("navChange")}qe.Zp.forEach(e=>(0,N.sp)(e,e=>(0,s.p)("ua",[e],void 0,this.featureName,this.ee),!0)),qe.qN.forEach(e=>{const t=(0,m.s)(e=>{(0,s.p)("ua",[e],void 0,this.featureName,this.ee)},500,{leading:!0});(0,N.sp)(e,t)}),f.gm.addEventListener("error",()=>{(0,s.p)("uaErr",[],void 0,t.K7.genericEvents,this.ee)},(0,N.jT)(!1,this.removeOnAbort.signal)),this.ee.on("open-xhr-start",(e,r)=>{p(e[1])||r.addEventListener("readystatechange",()=>{2===r.readyState&&(0,s.p)("uaXhr",[],void 0,t.K7.genericEvents,this.ee)},(0,N.jT)(void 0,this.removeOnAbort.signal))}),this.ee.on("fetch-start",e=>{e.length>=1&&!p(Te(e[0]))&&(0,s.p)("uaXhr",[],void 0,t.K7.genericEvents,this.ee)}),l.on("pushState-end",g),l.on("replaceState-end",g),window.addEventListener("hashchange",g,(0,N.jT)(!0,this.removeOnAbort.signal)),window.addEventListener("popstate",g,(0,N.jT)(!0,this.removeOnAbort.signal))}if(e.init.performance.resources.enabled&&f.gm.PerformanceObserver?.supportedEntryTypes.includes("resource")){new PerformanceObserver(e=>{e.getEntries().forEach(e=>{(0,s.p)("browserPerformance.resource",[e],void 0,this.featureName,this.ee)})}).observe({type:"resource",buffered:!0})}}d.some(e=>e)?this.importAggregator(e,()=>i.e(478).then(i.bind(i,8019))):this.deregisterDrain()}}var rt=i(2646);const nt=new Map;function it(e,t,r,n,i=!0,s){if("object"!=typeof t||!t||"string"!=typeof r||!r||"function"!=typeof t[r])return(0,h.R)(29);const o=function(e){return(e||oe.ee).get("logger")}(e),a=(0,ae.YM)(o,void 0,s),c=new rt.y(oe.P);c.level=n.level,c.customAttributes=n.customAttributes,c.autoCaptured=i;const d=t[r]?.[ae.Jt]||t[r];return nt.set(d,c),a.inPlace(t,[r],"wrap-logger-",()=>nt.get(d),void 0,!0),o}var st=i(1910);class ot extends y{static featureName=K.TZ;constructor(e){var t;super(e,K.TZ),t=e,(0,c.Y)(a.$9,(e,r)=>F(e,r,t),t),function(e){(0,c.Y)(a.Wb,(t,r,{customAttributes:n={},level:i=K.p_.INFO}={})=>{it(e.ee,t,r,{customAttributes:n,level:i},!1,e)},e)}(e),Q(e);const r=this.ee;["log","error","warn","info","debug","trace"].forEach(t=>{(0,st.i)(f.gm.console[t]),it(r,f.gm.console,t,{level:"log"===t?"info":t},void 0,e)}),this.ee.on("wrap-logger-end",function([e],t,n,i=[]){const{level:s,customAttributes:o,autoCaptured:a}=this;i.forEach(t=>{(0,W.R)(r,e,o,s,a,t)})}),this.importAggregator(e,()=>i.e(478).then(i.bind(i,5288)))}}new A({features:[Pe,E,_,Ve,Ge,P,ee,tt,ot,Ze],loaderType:"spa"})})()})();</script>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        
        
        <meta name="referrer" content="unsafe-url" />

        
           
                <title>Stanford Social Innovation Review: Informing and Inspiring Leaders of Social Change</title>
            

            
                <meta property="og:type"   content="website" />
            

            
                <link rel="canonical" href="https://ssir.org"/>
                <meta property="og:url" content="https://ssir.org" />
            

            <meta name="twitter:card" content="summary">

            

            

            
        
        <script
                type="text/javascript"
                src="https://app.termly.io/resource-blocker/02bad1f4-ec36-494f-b5c0-bff7d402d296?autoBlock=on"
                ></script>

        <script>
        dataLayer = [{
            "subscriberLoggedIn": "Guests",
            
            
            
            
            
            
            
            
            "gtmStatus": "prod",
            "paywallStatus": ""
        }];
        </script>

        <!-- Google Tag Manager -->
        <script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src='https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);})(window,document,'script','dataLayer','GTM-TNTS5VB');</script>
        <!-- End Google Tag Manager -->

        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-8XRL7X45FT"></script>
        <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());

        gtag('config', 'G-8XRL7X45FT');
        </script>
        

        <meta name="msvalidate.01" content="5374F42B10367E0CE1CD03BC47AFB94C" /> 
        
        <link rel="icon" type="image/png" href="/assets/favicon/favicon-96x96.png" sizes="96x96" />
        <link rel="icon" type="image/svg+xml" href="/assets/favicon/favicon.svg" />
        <link rel="shortcut icon" href="/assets/favicon/favicon.ico" />
        <link rel="apple-touch-icon" sizes="180x180" href="/assets/favicon/apple-touch-icon.png" />
        <link rel="manifest" href="/assets/favicon/site.webmanifest" />

        <meta property="fb:app_id" content="1566208430350173" />
        
        <!-- Global CSS -->
        <link rel="preload" href="https://fonts.googleapis.com/css?family=Titillium+Web" as="style" onload="this.onload=null;this.rel='stylesheet'">
    <noscript><link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Titillium+Web"></noscript>

    <link rel="preload" href="/css/fonts.css?v=2026-02-12" as="style" onload="this.onload=null;this.rel='stylesheet'">
    <noscript><link rel="stylesheet" href="/css/fonts.css?v=2026-02-12"></noscript>

    
    <link rel="preload" href="/css/style.css?v=2026-03-09-01" as="style" onload="this.onload=null;this.rel='stylesheet'"/>
    <!-- Preload CSS -->
    <link rel="preload" href="/css/style_async.css?v=2026-03-27" as="style" onload="this.onload=null;this.rel='stylesheet'">
    <noscript><link rel="stylesheet" href="/css/style_async.css?v=2026-03-27"></noscript>

    <link rel="stylesheet" type="text/css" href="https://ssir.org/minify/16daa1c2250228f4361e2e7aac8b293804fec255.1527098460.css" rel="none" onload="this.onload=null;this.rel='stylesheet'" />
    <noscript>
    <link rel="stylesheet" href="https://ssir.org/minify/4fa6bb2191ea6a584b17c00306b4dfea168d8872.1773316872.css" />
    </noscript>



    <link rel="preload" href="/css/style_sponsored_report.css?v=2026-02-12" as="style" onload="this.onload=null;this.rel='stylesheet'">
    <noscript><link rel="stylesheet" href="/css/style_sponsored_report.css?v=2026-02-12"></noscript>






        
        

        <!-- BEGIN GOOGLE SEARCH CODE -->
        <script>
          (function() {
            var cx = '008920298281052476179:dtxgwiwqvhc'; // NOT 011699993154810244094:4tb_2axeqmu
            var gcse = document.createElement('script');
            gcse.type = 'text/javascript';
            gcse.async = true;
            gcse.src = (document.location.protocol == 'https:' ? 'https:' : 'http:') +
                '//cse.google.com/cse.js?cx=' + cx;
            var s = document.getElementsByTagName('script')[0];
            s.parentNode.insertBefore(gcse, s);
          })();
        </script>
        <!-- END GOOGLE SEARCH CODE -->

        <meta name="Copyright" content="(c) 2005-2026" />

        
        <link rel="alternate" type="application/rss+xml" title="SSIR Articles" href="https://ssir.org/site/rss_2.0" />
        <link rel="alternate" type="application/rss+xml" title="SSIR Analysis and Opinion Blog" href="https://ssir.org/blog/rss" />
        <link rel="apple-touch-icon-precomposed" HREF="https://images.ssireview.org/apple-touch-icon.png" />

        
        <script async src="https://securepubads.g.doubleclick.net/tag/js/gpt.js"></script>
        <script>
            window.googletag = window.googletag || {cmd: []};
            googletag.cmd.push(function() {
                var mappingTop = googletag.sizeMapping()
                                           .addSize([758, 480], [728, 90])
                                           .addSize([0, 0], [320, 50]) // All viewports < 640x480
                                           .build();
                
                googletag.defineSlot('/22062841453/SSIR_728x90', [[320, 50], [728, 90]], 'div-gpt-ad-1636034642111-0').defineSizeMapping(mappingTop).addService(googletag.pubads());
                googletag.defineSlot('/22062841453/SSIR_HighImpact', [[600, 250], [468, 60], [300, 250]], 'div-gpt-ad-1646354848641-0').addService(googletag.pubads());
                
googletag.defineSlot('/22062841453/SSIR_612x105', [612, 105], 'div-gpt-ad-1596831006998-0').addService(googletag.pubads());

                        	googletag.defineSlot('/22062841453/300x250ATF', [300, 250], 'div-gpt-ad-1596666215664-0').addService(googletag.pubads());
                            googletag.defineSlot('/22062841453/SSIR_300x250BTF', [300, 250], 'div-gpt-ad-1596665496550-0').addService(googletag.pubads());
                        
                googletag.pubads().enableSingleRequest();
                googletag.enableServices();
            });
        </script>
        
        <script data-ad-client="ca-pub-8471827283965476" async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>
        
        
    </head>
    <body>
        <div id="refreshed_csrf_token" style="display: none;"><input type="hidden" name="csrf_token" value="d1ab0e2b5ec1c067ca2c8f9bced8fb1695d044ef" data-updated="true" /></div>
        <script async>
            function fetchAndInjectHTML(url, scriptElement = null, shouldInject = true, callback = null) {
                return (async () => {
                    try {
                        const response = await fetch(url);
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        const htmlContent = await response.text();

                        if (shouldInject && scriptElement) {
                            const container = document.createElement('div');
                            container.innerHTML = htmlContent;
                            scriptElement.insertAdjacentElement('afterend', container);

                            // Remove the script tag after injection
                            scriptElement.remove();
                        }
                        if (callback) {callback();}

                        return htmlContent; // Always return the HTML content
                    } catch (error) {
                        console.error('Error fetching and injecting HTML:', error);
                        return null; // Return null in case of an error
                    }
                })();
            }
        </script>
        <main>
<!-- Google Tag Manager (noscript) -->
<noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-TNTS5VB" height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
<!-- End Google Tag Manager (noscript) -->
        
        <script defer>fetchAndInjectHTML('/defer-load/cookie-reset', document.currentScript);</script>
        <div class="wrapper">
            <div class="offset-nav">
        <header class="offset-nav-head">
            <a href="https://ssir.org/" class="home">
                <i class="ico ico-home" aria-hidden="true"></i>
            </a>
            <a href="#" class="close">
                <i class="ico ico-close-grey" aria-hidden="true"></i>
            </a>
        </header><!-- /.offset-nav-head -->
        
        <div class="offset-nav-body">
            <nav class="nav-alt">
                <ul>
                    <li class="has-submenu first">
                        <a href="#" class="uppercased">Social Issues <span>Education, Health, Security, etc.</span></a>
                        <div class="offset-nav-dropdown">
                            <header class="offset-nav-dropdown-head">
                                <a href="#" class="back">
                                    <i class="ico ico-back" aria-hidden="true"></i>
                                </a>
                                <a href="#" class="close">
                                    <i class="ico ico-close-grey" aria-hidden="true"></i>
                                </a>
                            </header><!-- /.offset-nav-dropdown-head -->
                            <div class="offset-nav-dropdown-body">
                                <ul>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/arts-culture">Arts & Culture</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/cities">Cities</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/civic-engagement">Civic Engagement</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/economic-development">Economic Development</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/education">Education</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/energy">Energy</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/environment">Environment</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/food">Food</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/health">Health</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/human-rights">Human Rights</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/security">Security</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/social-services">Social Services</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/water-sanitation">Water & Sanitation</a>
                                        </li>
                                    
                                </ul>
                            </div><!-- /.offset-nav-dropdown-body -->
                        </div><!-- /.nav-dropdown -->
                    </li>
                    <li class="has-submenu">
                        <a href="#" class="uppercased">Sectors <span>Government, Nonprofit, Business, etc.</span></a>
                        <div class="offset-nav-dropdown">
                            <header class="offset-nav-dropdown-head">
                                <a href="#" class="back">
                                    <i class="ico ico-back" aria-hidden="true"></i>
                                </a>
                                <a href="#" class="close">
                                    <i class="ico ico-close-grey" aria-hidden="true"></i>
                                </a>
                            </header><!-- /.offset-nav-dropdown-head -->
                            <div class="offset-nav-dropdown-body">
                                <ul>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/business">Business</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/foundations">Foundations</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/government">Government</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/nonprofits-ngos">Nonprofits & NGOs</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/social-enterprise">Social Enterprise</a>
                                        </li>
                                    
                                </ul>
                            </div><!-- /.offset-nav-dropdown-body -->
                        </div><!-- /.nav-dropdown -->
                    </li>
                    <li class="has-submenu">
                        <a href="#" class="uppercased">Solutions <span>Advocacy, Funding, Leadership, etc.</span></a>
                        <div class="offset-nav-dropdown">
                            <header class="offset-nav-dropdown-head">
                                <a href="#" class="back">
                                    <i class="ico ico-back" aria-hidden="true"></i>
                                </a>
                                <a href="#" class="close">
                                    <i class="ico ico-close-grey" aria-hidden="true"></i>
                                </a>
                            </header><!-- /.offset-nav-dropdown-head -->
                            <div class="offset-nav-dropdown-body">
                                <ul>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/advocacy">Advocacy</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/collaboration">Collaboration</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/design-thinking">Design Thinking</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/governance">Governance</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/impact-investing">Impact Investing</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/leadership">Leadership</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/measurement-evaluation">Measurement & Evaluation</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/organizational-development">Organizational Development</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/philanthropy-funding">Philanthropy & Funding</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/scaling">Scaling</a>
                                        </li>
                                    
                                        <li>
                                            <a href="https://ssir.org/topics/category/technology">Technology</a>
                                        </li>
                                    
                                </ul>
                            </div><!-- /.offset-nav-dropdown-body -->
                        </div><!-- /.nav-dropdown -->
                    </li>
                    <li class="has-submenu">
                        <a href="#" class="uppercased">Magazine</a>
                        <div class="offset-nav-dropdown">
                            <header class="offset-nav-dropdown-head">
                                <a href="#" class="back">
                                    <i class="ico ico-back" aria-hidden="true"></i>
                                </a>
                                <a href="#" class="close">
                                    <i class="ico ico-close-grey" aria-hidden="true"></i>
                                </a>
                            </header><!-- /.offset-nav-dropdown-head -->
                            <div class="offset-nav-dropdown-body">
                                <ul>
                                    <li>
                                        <!-- Using current_time to bypass cached redirect -->
                                        <a href="https://ssir.org/issue/spring-2026">Current Issue</a>
                                    </li>
                                    <li>
                                        <a href="https://ssir.org/issue">All Issues</a>
                                    </li>
                                    <li>
                                        <a href="https://ssir.org/supplement/magazine-supplements">Sponsored Supplements</a>
                                    </li>
                                </ul>
                            </div><!-- /.offset-nav-dropdown-body -->
                        </div><!-- /.nav-dropdown -->
                    </li>
                    <li>
                        <a href="https://ssir.org/supplement/in-depth-series" class="uppercased">In-Depth Series</a>
                    </li>
                    <li>
                        <a href="https://ssir.org/podcasts" class="uppercased">Podcasts</a>
                    </li>
                    <li>
                        <a href="https://ssir.org/webinars" class="uppercased">Webinars</a>
                    </li>
                    <li>
                        <a href="https://ssir.org/events" class="uppercased">Events</a>
                    </li>
                    <li>
                        <a href="https://ssir.org/book_reviews" class="uppercased">Books</a>
                    </li>
                    <li>
                        <a href="https://ssir.org/videos" class="uppercased">Videos</a>
                    </li>
                    <li>
                        <a href="https://ssir.org/email" class="uppercased">Newsletter</a>
                    </li>
                    <li>
                        <a href="https://ssir.org/ssir-global" class="uppercased">Global Editions</a>
                    </li>
                    <li class="has-submenu">
                        <a href="https://ssir.org/about/overview" class="uppercased">About Us</a>
                        <div class="offset-nav-dropdown">
                            <header class="offset-nav-dropdown-head">
                                <a href="#" class="back">
                                    <i class="ico ico-back" aria-hidden="true"></i>
                                </a>
                                <a href="#" class="close">
                                    <i class="ico ico-close-grey" aria-hidden="true"></i>
                                </a>
                            </header><!-- /.offset-nav-dropdown-head -->
                            <div class="offset-nav-dropdown-body">
                                <ul>
                                    	<li>
		<a href="https://ssir.org/about/overview">About <em>SSIR</em></a>
	</li>
	<li>
		<a href="http://pacscenter.stanford.edu/">Stanford PACS</a>
	</li>
	<li>
		<a href="https://ssir.org/about/contact">Contact Us</a>
	</li>
	<li>
		<a href="https://ssir.org/about/submission-guidelines">Submissions</a>
	</li>
	<li>
		<a href="https://ssir.org/advertising">Advertise</a>
	</li>
	<li>
		<a href="https://ssir.org/help">Help</a>
	</li>
	<li>
		<a href="https://ssir.org/reprints-permissions">Reprints</a>
	</li>
	<li>
		<a href="http://www.stanford.edu/site/terms.html">Terms of Use</a>
	</li>
	<li>
		<a href="https://ssir.org/privacy-policy">Privacy</a>
	</li>
		<li>
		<a href="https://www.stanford.edu/site/accessibility/">Accessibility</a>
	</li>
	<li>
		<a href="https://ssir.org/site-map">Site Map</a>
	</li>

                                </ul>
                            </div><!-- /.offset-nav-dropdown-body -->
                        </div><!-- /.nav-dropdown -->
                    </li>
                    <li>
                        <a href="https://ssir.org/donate" class="btn-donate uppercased" title="Donate to SSIR">Donate</a>
                    </li>
                </ul>
            </nav><!-- /.nav-alt -->
        </div><!-- /.offset-nav-body -->
    </div><!-- /.offset-nav -->
    
    <div class="widget-sticky">
    <nav class="pigeon-widget-status"></nav>
    <div class="widget-sticky-nav-mobile">
        <div class="search">
            <button class="btn-search" aria-label="Search">
                <i class="ico ico-search" aria-hidden="true"></i>
            </button>
        </div><!-- /.search -->
        <a href="#" class="nav-trigger" aria-label="Open menu">
            <span class="bar"></span>
            <span class="bar"></span>
            <span class="bar"></span>
        </a>
    </div><!-- /.widget-sticky-nav-mobile -->
</div><!-- /.widget-sticky -->
<div class="search-lightbox" id="search-lightbox" aria-hidden="true">
    <div class="search-lightbox-backdrop"></div>
    <div class="search-lightbox-content">
        <button type="button" class="search-lightbox-close" aria-label="Close search">
            <span aria-hidden="true">&times;</span>
        </button>
        <h3 class="search-lightbox-title">Search <em>SSIR</em></h3>
        <div id="search-container"><gcse:searchbox-only resultsUrl="https://ssir.org/search/results"></gcse:searchbox-only></div>
    </div>
</div>
    <div id="ad-top-container" class="shell">
        <div class="header-top-inner">
            <div class="banner banner-large">
                <div class="desktop-view">
                    
                    <div id='div-gpt-ad-1636034642111-0' style='min-width: 320px; min-height: 50px; text-align: center;'>
                        <script>
                            googletag.cmd.push(function() {googletag.display('div-gpt-ad-1636034642111-0');});
                        </script>
                    </div>
                </div>
            </div><!-- /.banner banner-large -->
        </div><!-- /.header-top-inner -->
    </div><!-- /.shell -->
    <header class="header">
        <div class="header-bottom">
            <div class="shell">
                <div class="header-bottom-inner">
                    <a href="https://ssir.org/" class="logo" aria-label="SSIR Homepage">
                        <img src="/images/resources/SSIR_Logo.RGB_Black_tagline_984x234.png" height="120" width="492" alt="Stanford Social Innovation Review Logo">
                    </a>
                    
                    <div class="header-subscribe">
    
        
            
                <a href="https://subscriber.ssir.org"><img src="https://ssir.org/images/jcogs_img/cache/spring-2026-770x1009_-_abcdef_-_8cb579eb8dd0e5458f305dc2efdb68c73d1f384a.webp" alt="" id="img_rotate45_b" height="127" width="97"/></a>
            
        
        
    

    
    <ul>
        <a href="https://ssir.org/donate" class="btn-donate uppercased">Donate</a>
    </ul>
</div><!-- /.header-subscribe -->
                </div><!-- /.header-bottom-inner -->
            </div><!-- /.shell -->
        </div><!-- /.header-bottom -->
    </header><!-- /.header -->
    <div class="nav">
        <div class="shell">
            <div class="nav-inner">
                <ul>
                    <li class="first">
                        <a href="#">Social Issues</a>
                        <div class="nav-dropdown multiple">
                            <ul>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/arts-culture">Arts & Culture</a>
                                    </li>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/cities">Cities</a>
                                    </li>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/civic-engagement">Civic Engagement</a>
                                    </li>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/economic-development">Economic Development</a>
                                    </li>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/education">Education</a>
                                    </li>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/energy">Energy</a>
                                    </li>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/environment">Environment</a>
                                    </li></ul> <ul>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/food">Food</a>
                                    </li>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/health">Health</a>
                                    </li>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/human-rights">Human Rights</a>
                                    </li>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/security">Security</a>
                                    </li>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/social-services">Social Services</a>
                                    </li>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/water-sanitation">Water & Sanitation</a>
                                    </li></ul>
                        </div><!-- /.nav-dropdown -->
                    </li>
                    <li>
                        <a href="#">Sectors</a>
                        <div class="nav-dropdown">
                            <ul>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/business">Business</a>
                                    </li>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/foundations">Foundations</a>
                                    </li>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/government">Government</a>
                                    </li>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/nonprofits-ngos">Nonprofits & NGOs</a>
                                    </li>
                                
                                    <li>
                                        <a href="https://ssir.org/topics/category/social-enterprise">Social Enterprise</a>
                                    </li>
                                
                            </ul>
                        </div><!-- /.nav-dropdown -->
                    </li>
                    <li>
                        <a href="#">Solutions</a>
                        <div class="nav-dropdown multiple">
                            <ul>
                            
                                <li>
                                    <a href="https://ssir.org/topics/category/advocacy">Advocacy</a>
                                </li>
                            
                                <li>
                                    <a href="https://ssir.org/topics/category/collaboration">Collaboration</a>
                                </li>
                            
                                <li>
                                    <a href="https://ssir.org/topics/category/design-thinking">Design Thinking</a>
                                </li>
                            
                                <li>
                                    <a href="https://ssir.org/topics/category/governance">Governance</a>
                                </li>
                            
                                <li>
                                    <a href="https://ssir.org/topics/category/impact-investing">Impact Investing</a>
                                </li>
                            
                                <li>
                                    <a href="https://ssir.org/topics/category/leadership">Leadership</a>
                                </li></ul> <ul>
                            
                                <li>
                                    <a href="https://ssir.org/topics/category/measurement-evaluation">Measurement & Evaluation</a>
                                </li>
                            
                                <li>
                                    <a href="https://ssir.org/topics/category/organizational-development">Organizational Development</a>
                                </li>
                            
                                <li>
                                    <a href="https://ssir.org/topics/category/philanthropy-funding">Philanthropy & Funding</a>
                                </li>
                            
                                <li>
                                    <a href="https://ssir.org/topics/category/scaling">Scaling</a>
                                </li>
                            
                                <li>
                                    <a href="https://ssir.org/topics/category/technology">Technology</a>
                                </li></ul>
                        </div><!-- /.nav-dropdown multiple -->
                    </li>
                    <li class="bordered">
                        <a href="#">magazine</a>
                        <div class="nav-dropdown">
                            <ul>
                                <li>
                                    <!-- Using current_time to bypass cached redirect -->
                                    <a href="https://ssir.org/issue/spring-2026">Current Issue</a>
                                </li>
                                <li>
                                    <a href="https://ssir.org/issue">All Issues</a>
                                </li>
                                <li>
                                    <a href="https://ssir.org/supplement/magazine-supplements">Sponsored Supplements</a>
                                </li>
                            </ul>
                        </div><!-- /.nav-dropdown -->
                    </li>
                    <li>
                        <a href="https://ssir.org/ssir-global">Global Editions</a>
                    </li>
                    <li>
                        <a href="#">more</a>
                        <div class="nav-dropdown">
                            <ul>
                                <li>
                                    <a href="https://ssir.org/supplement/in-depth-series">In-Depth Series</a>
                                </li>
                                <li>
                                    <a href="https://ssir.org/podcasts">Podcasts</a>
                                </li>
                                <li>
                                    <a href="https://ssir.org/webinars">Webinars</a>
                                </li>
                                <li>
                                    <a href="https://ssir.org/events">Events</a>
                                </li>
                                <li>
                                    <a href="https://ssir.org/book_reviews">Books</a>
                                </li>
                                <li>
                                    <a href="https://ssir.org/videos">Videos</a>
                                </li>
                                <li>
                                    <a href="https://ssir.org/email">Newsletter</a>
                                </li>
                                <li>
                                    <a href="http://pacscenter.stanford.edu/">Stanford PACS</a>
                                </li>
                                <li>
                                    <a href="https://ssir.org/about/submission_guidelines">Submission Guidelines</a>
                                </li>
                                <li>
                                    <a href="https://ssir.org/about/overview">About Us</a>
                                </li>
                            </ul>
                        </div><!-- /.nav-dropdown -->
                    </li>
                </ul>
                <div class="search">
                    <button class="btn-search" aria-label="Search">
                        <i class="ico ico-search" aria-hidden="true"></i>
                    </button>
                </div><!-- /.search -->
                <div class="socials">
                    <ul>
                        <li>
                            <a href="https://www.linkedin.com/company/stanford-social-innovation-review" aria-label="SSIR Linkedin">
                                <i class="ico-linkedin"></i>
                            </a>
                        </li>
                        <li>
                            <a href="https://www.instagram.com/stanford.ssir/" aria-label="SSIR Instagram">
                                <i class="ico-instagram"></i>
                            </a>
                        </li>
                        <li>
                            <a href="https://bsky.app/profile/ssir.org" aria-label="SSIR Bluesky">
                                <img src="/css/images/ico-bluesky-header.svg" alt="Bluesky Logo" width="24px"/>
                            </a>
                        </li>
                        <li>
                            <a href="https://ssir.org/site/rss_2.0" aria-label="SSIR RSS">
                                <i class="ico-rss"></i>
                            </a>
                        </li>
                    </ul>
                </div><!-- /.socials -->
            </div><!-- /.nav-inner -->
        </div><!-- /.shell -->
    </div><!-- /.nav -->

            
            
            <section class="section-homepage section-intro ">
    <div class="shell">
        <div class="section-inner">
            
            <div class="section-image"> 
                
                <a href="https://ssir.org/articles/entry/grant-writing-infrastructure" data-test="https://ssir.org/images/homepage/spring-2026-viewpoint-grant-hp-737x578.jpg">
                    
                        
                            <img src="https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-grant-hp-737x578_-_abcdef_-_8b691b5e2242237e2703380d1291bf3a36461912.webp" width="737" height="578" alt="" srcset="https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-grant-hp-737x578_-_abcdef_-_8b691b5e2242237e2703380d1291bf3a36461912_292w.webp 292w, https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-grant-hp-737x578_-_abcdef_-_8b691b5e2242237e2703380d1291bf3a36461912_383w.webp 383w, https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-grant-hp-737x578_-_abcdef_-_8b691b5e2242237e2703380d1291bf3a36461912_582w.webp 582w, https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-grant-hp-737x578_-_abcdef_-_8b691b5e2242237e2703380d1291bf3a36461912_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-grant-hp-737x578_-_abcdef_-_8b691b5e2242237e2703380d1291bf3a36461912.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), (max-width: 996px) calc((100vw - 40px) * .4), (max-width: 1994px) 383px, 737px" />
                        
                    
                </a>
                  
  	
	
  

  
  
  
            </div><!-- /.section-image -->

            <div class="section-body">
                
                <h5>




    
        
            
                
                    <a href="https://ssir.org/topics/category/nonprofits-ngos">Nonprofits & NGOs</a>
                

                

                
            
        

        
    
</h5>

                <h1>
                    <a href="https://ssir.org/articles/entry/grant-writing-infrastructure"> Nonprofits Need Grant Writing Infrastructure </a>
                    
                    <span>
                        
                            

	By <a class="author" href="https://ssir.org/bios/matt-leighty">Matt Leighty</a> & <a class="author" href="https://ssir.org/bios/emily-joseph">Emily Joseph</a>

                        
                        
                    </span>
                    
                </h1>

                <div class="feature-summary">
                    
                        <p>Grant writing is the bridge between vision and resources. Until funders invest in it, grassroots organizations will remain underfunded.<br></p>
                        
                        
                        
                        
                        
                        
                        
                        
                        
                    
                </div>
            </div><!-- /.section-body -->
        </div><!-- /.section-inner -->
        
    </div><!-- /.shell -->
</section><!-- /.section-intro -->
        

            <div class="main">
                <div class="shell">
                    

            <div class="main-inner clearfix">

				

				<section class="content">
                    



 




<script type="application/ld+json">
{
  "@context": "http://schema.org",
  "@type": "WebSite",
  "url": "https://ssir.org/",
  "potentialAction": {
    "@type": "SearchAction",
    "target": "https://ssir.org/search/results?q={search_term_string}",
    "query-input": "required name=search_term_string"
  }
}
</script>



    
    

        
        
    



    
    
        
        <section class="section-featured">
        
        <article class="article featured-article">
    <div class="article-body">
        
        <div class="article-image">
            <a href="https://ssir.org/articles/entry/sigelman-education-for-thriving-careers">

                
                    
                        <img src="https://ssir.org/images/jcogs_img/cache/education-thriving-careers-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95.webp" width="737" height="578" alt="" srcset="https://ssir.org/images/jcogs_img/cache/education-thriving-careers-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_187w.webp 187w, https://ssir.org/images/jcogs_img/cache/education-thriving-careers-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_290w.webp 290w, https://ssir.org/images/jcogs_img/cache/education-thriving-careers-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_420w.webp 420w, https://ssir.org/images/jcogs_img/cache/education-thriving-careers-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_543w.webp 543w, https://ssir.org/images/jcogs_img/cache/education-thriving-careers-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/education-thriving-careers-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), (max-width: 996px) calc((100vw - 40px) * .2835), (max-width: 1994px) 290px, (max-width: 2991px) 543px, 737px" />
                    
                
            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->

        <div class="article-entry">
            
	<h5 class="sponsored">
		<span>sponsored</span>
	</h5>

            <h5>




    
        
            
                
                    <a href="https://ssir.org/topics/category/education">Education</a>
                

                

                
            
        

        
    
</h5>

            <h4>
                <a href="https://ssir.org/articles/entry/sigelman-education-for-thriving-careers">Education for Thriving Careers</a>
                <script defer>fetchAndInjectHTML('/defer-load/edit-this/19981', document.currentScript);</script>
            </h4>

            
            <div class="meta">
                <p>
                    
                        

	By <a class="author" href="https://ssir.org/bios/matt-sigelman">Matt Sigelman</a>

                    

                    
                </p>
            </div><!-- /.meta -->
            

            <div class="feature-summary">
                
                    
                    
                    <p>What the research says about education, jobs, AI, and what students will need to succeed as future workers and citizens. </p>
                    
                    
                    
                    
                    
                    
                    
                
            </div>
        </div><!-- /.article-entry -->
        
    </div><!-- /.article-body -->
</article><!-- /.featured-article -->

        
    
        
        <article class="article featured-article">
    <div class="article-body">
        
        <div class="article-image">
            <a href="https://ssir.org/articles/entry/messano-what-are-schools-for-now">

                
                    
                        <img src="https://ssir.org/images/jcogs_img/cache/what-are-schools-for-now-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95.webp" width="737" height="578" alt="" srcset="https://ssir.org/images/jcogs_img/cache/what-are-schools-for-now-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_187w.webp 187w, https://ssir.org/images/jcogs_img/cache/what-are-schools-for-now-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_290w.webp 290w, https://ssir.org/images/jcogs_img/cache/what-are-schools-for-now-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_420w.webp 420w, https://ssir.org/images/jcogs_img/cache/what-are-schools-for-now-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_543w.webp 543w, https://ssir.org/images/jcogs_img/cache/what-are-schools-for-now-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/what-are-schools-for-now-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), (max-width: 996px) calc((100vw - 40px) * .2835), (max-width: 1994px) 290px, (max-width: 2991px) 543px, 737px" />
                    
                
            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->

        <div class="article-entry">
            
	<h5 class="sponsored">
		<span>sponsored</span>
	</h5>

            <h5>




    
        
            
                
                    <a href="https://ssir.org/topics/category/education">Education</a>
                

                

                
            
        

        
    
</h5>

            <h4>
                <a href="https://ssir.org/articles/entry/messano-what-are-schools-for-now">What Are Schools For Now?</a>
                <script defer>fetchAndInjectHTML('/defer-load/edit-this/19979', document.currentScript);</script>
            </h4>

            
            <div class="meta">
                <p>
                    
                        

	By <a class="author" href="https://ssir.org/bios/frances_messano">Frances Messano</a>

                    

                    
                        <span>
                            <a title="Comments" href="https://ssir.org/articles/entry/messano-what-are-schools-for-now#comments">1 <i class="ico ico-comment" aria-hidden="true"></i></a>
                        </span>
                    
                </p>
            </div><!-- /.meta -->
            

            <div class="feature-summary">
                
                    
                    
                    <p>Reimagining education for a changing world</p>
                    
                    
                    
                    
                    
                    
                    
                
            </div>
        </div><!-- /.article-entry -->
        
    </div><!-- /.article-body -->
</article><!-- /.featured-article -->

        
        </section><!-- /.section-featured -->
        
    

    <div>
        <a href="https://ssir.org/donate" class="btn-donate uppercased inline-donation-btn" title="Donate to SSIR">Donate to SSIR</a>
    </div>

    
    
        
        <section class="section-articles-inline section-homepage">
            <div class="section-body">
        

        <article class="article article-inline">
    <div class="article-body">
        
        <div class="article-image">
            <a href="https://ssir.org/book_reviews/entry/beyond-belief-helen-pearson">

                
                    
                        <img src="https://ssir.org/images/jcogs_img/cache/beyond-belief-cover-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95.webp" width="737" height="578" alt="" srcset="https://ssir.org/images/jcogs_img/cache/beyond-belief-cover-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_187w.webp 187w, https://ssir.org/images/jcogs_img/cache/beyond-belief-cover-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_290w.webp 290w, https://ssir.org/images/jcogs_img/cache/beyond-belief-cover-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_420w.webp 420w, https://ssir.org/images/jcogs_img/cache/beyond-belief-cover-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_543w.webp 543w, https://ssir.org/images/jcogs_img/cache/beyond-belief-cover-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/beyond-belief-cover-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), (max-width: 996px) calc((100vw - 40px) * .1953), (max-width: 1994px) 187px, (max-width: 2991px) 420px, 737px" />
                    
                
            </a>
              
  	
	<div class="article-label"> 
	  <i class="fa fa-excerpt" aria-hidden="true"></i><span class="uppercased">Excerpt</span>
	</div>
  	
	
  

  
  
  
        </div><!-- /.article-image -->

        <div class="article-entry">
            
            <h5>




    
        
            
                
                    <a href="https://ssir.org/topics/category/measurement-evaluation">Measurement & Evaluation</a>
                

                

                
            
        

        
    
</h5>

            <h4>
                <a href="https://ssir.org/book_reviews/entry/beyond-belief-helen-pearson">The Randomista Revolt</a>
                <script defer>fetchAndInjectHTML('/defer-load/edit-this/19985', document.currentScript);</script>
            </h4>

            
            <div class="meta">
                <p>
                    
                        

	By <a class="author" href="https://ssir.org/bios/helen-pearson">Helen Pearson</a>

                    

                    
                </p>
            </div><!-- /.meta -->
            

            <div class="feature-summary">
                
                    <p>An excerpt from <em>Beyond Belief</em> on building the evidence revolution in Washington</p>
                    
                    
                    
                    
                    
                    
                    
                    
                    
                
            </div>
        </div><!-- /.article-entry -->
        
    </div><!-- /.article-body -->
</article><!-- /.featured-article -->

        
        
    
        

        <article class="article article-inline">
    <div class="article-body">
        
        <div class="article-image">
            <a href="https://ssir.org/articles/entry/supercharging-network-intelligence">

                
                    
                        <img src="https://ssir.org/images/jcogs_img/cache/supercharging-network-intelligence-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95.webp" width="737" height="578" alt="" srcset="https://ssir.org/images/jcogs_img/cache/supercharging-network-intelligence-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_187w.webp 187w, https://ssir.org/images/jcogs_img/cache/supercharging-network-intelligence-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_290w.webp 290w, https://ssir.org/images/jcogs_img/cache/supercharging-network-intelligence-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_420w.webp 420w, https://ssir.org/images/jcogs_img/cache/supercharging-network-intelligence-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_543w.webp 543w, https://ssir.org/images/jcogs_img/cache/supercharging-network-intelligence-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/supercharging-network-intelligence-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), (max-width: 996px) calc((100vw - 40px) * .1953), (max-width: 1994px) 187px, (max-width: 2991px) 420px, 737px" />
                    
                
            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->

        <div class="article-entry">
            
            <h5>




    
        
            
                
                    <a href="https://ssir.org/topics/category/technology">Technology</a>
                

                

                
            
        

        
    
</h5>

            <h4>
                <a href="https://ssir.org/articles/entry/supercharging-network-intelligence">Supercharging Network Intelligence</a>
                <script defer>fetchAndInjectHTML('/defer-load/edit-this/19976', document.currentScript);</script>
            </h4>

            
            <div class="meta">
                <p>
                    
                        

	By <a class="author" href="https://ssir.org/bios/nikolaj-moesgaard">Nikolaj Moesgaard</a> & <a class="author" href="https://ssir.org/bios/gueliz-berfin-kolda">Güliz Berfin Koldaş</a>

                    

                    
                </p>
            </div><!-- /.meta -->
            

            <div class="feature-summary">
                
                    
                    
                    <p>How network organizations can use AI to better understand and support their members in real time.</p>
                    
                    
                    
                    
                    
                    
                    
                
            </div>
        </div><!-- /.article-entry -->
        
    </div><!-- /.article-body -->
</article><!-- /.featured-article -->

        
        
    
        

        <article class="article article-inline">
    <div class="article-body">
        
        <div class="article-image">
            <a href="https://ssir.org/articles/entry/airborne-disease-paradigm-shift">

                
                    
                        <img src="https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-paradigm-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95.webp" width="737" height="578" alt="" srcset="https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-paradigm-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_187w.webp 187w, https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-paradigm-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_290w.webp 290w, https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-paradigm-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_420w.webp 420w, https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-paradigm-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_543w.webp 543w, https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-paradigm-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-paradigm-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), (max-width: 996px) calc((100vw - 40px) * .1953), (max-width: 1994px) 187px, (max-width: 2991px) 420px, 737px" />
                    
                
            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->

        <div class="article-entry">
            
            <h5>




    
        
            
                
                    <a href="https://ssir.org/topics/category/health">Health</a>
                

                

                
            
        

        
    
</h5>

            <h4>
                <a href="https://ssir.org/articles/entry/airborne-disease-paradigm-shift">The New Paradigm of Airborne Disease</a>
                <script defer>fetchAndInjectHTML('/defer-load/edit-this/19854', document.currentScript);</script>
            </h4>

            
            <div class="meta">
                <p>
                    
                        

	By <a class="author" href="https://ssir.org/bios/david-carel">David Carel</a>

                    

                    
                        <span>
                            <a title="Comments" href="https://ssir.org/articles/entry/airborne-disease-paradigm-shift#comments">1 <i class="ico ico-comment" aria-hidden="true"></i></a>
                        </span>
                    
                </p>
            </div><!-- /.meta -->
            

            <div class="feature-summary">
                
                    <p>The COVID-19 pandemic shattered established views of airborne disease. Although it represents a paradigm shift in public health, the field has yet to catch up.</p>
                    
                    
                    
                    
                    
                    
                    
                    
                    
                
            </div>
        </div><!-- /.article-entry -->
        
    </div><!-- /.article-body -->
</article><!-- /.featured-article -->

        
        </div><!-- /.section-body -->
        </section><!-- /.section-articles-inline -->
        
        
            
        
    
        
        <section class="section-articles-inline section-homepage">
            <div class="section-body">
        

        <article class="article article-inline">
    <div class="article-body">
        
        <div class="article-image">
            <a href="https://ssir.org/articles/entry/civic-commons-democracy">

                
                    
                        <img src="https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-democracy-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95.webp" width="737" height="578" alt="" srcset="https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-democracy-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_187w.webp 187w, https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-democracy-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_290w.webp 290w, https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-democracy-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_420w.webp 420w, https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-democracy-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_543w.webp 543w, https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-democracy-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/spring-2026-viewpoint-democracy-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), (max-width: 996px) calc((100vw - 40px) * .1953), (max-width: 1994px) 187px, (max-width: 2991px) 420px, 737px" />
                    
                
            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->

        <div class="article-entry">
            
            <h5>




    
        
            
                
                    <a href="https://ssir.org/topics/category/civic-engagement">Civic Engagement</a>
                

                

                
            
        

        
    
</h5>

            <h4>
                <a href="https://ssir.org/articles/entry/civic-commons-democracy">Reimagining the Civic Commons</a>
                <script defer>fetchAndInjectHTML('/defer-load/edit-this/19852', document.currentScript);</script>
            </h4>

            
            <div class="meta">
                <p>
                    
                        

	By <a class="author" href="https://ssir.org/bios/alexa-bush">Alexa Bush</a> & <a class="author" href="https://ssir.org/bios/kyle-kutuchief">Kyle Kutuchief</a>

                    

                    
                        <span>
                            <a title="Comments" href="https://ssir.org/articles/entry/civic-commons-democracy#comments">1 <i class="ico ico-comment" aria-hidden="true"></i></a>
                        </span>
                    
                </p>
            </div><!-- /.meta -->
            

            <div class="feature-summary">
                
                    <p>After a decade of transforming public spaces, we are building trust and connection between Americans.</p>
                    
                    
                    
                    
                    
                    
                    
                    
                    
                
            </div>
        </div><!-- /.article-entry -->
        
    </div><!-- /.article-body -->
</article><!-- /.featured-article -->

        
        
    
        

        <article class="article article-inline">
    <div class="article-body">
        
        <div class="article-image">
            <a href="https://ssir.org/articles/entry/new-leaders-slow-integration-success">

                
                    
                        <img src="https://ssir.org/images/jcogs_img/cache/spring-2026-research-treating-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95.webp" width="737" height="578" alt="" srcset="https://ssir.org/images/jcogs_img/cache/spring-2026-research-treating-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_187w.webp 187w, https://ssir.org/images/jcogs_img/cache/spring-2026-research-treating-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_290w.webp 290w, https://ssir.org/images/jcogs_img/cache/spring-2026-research-treating-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_420w.webp 420w, https://ssir.org/images/jcogs_img/cache/spring-2026-research-treating-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_543w.webp 543w, https://ssir.org/images/jcogs_img/cache/spring-2026-research-treating-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/spring-2026-research-treating-hp-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), (max-width: 996px) calc((100vw - 40px) * .1953), (max-width: 1994px) 187px, (max-width: 2991px) 420px, 737px" />
                    
                
            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->

        <div class="article-entry">
            
            <h5>




    
        
            
                
                    <a href="https://ssir.org/topics/category/organizational-development">Organizational Development</a>
                

                

                
            
        

        
    
</h5>

            <h4>
                <a href="https://ssir.org/articles/entry/new-leaders-slow-integration-success">Treating New Hires as Strangers</a>
                <script defer>fetchAndInjectHTML('/defer-load/edit-this/19865', document.currentScript);</script>
            </h4>

            
            <div class="meta">
                <p>
                    
                        

	By <a class="author" href="https://ssir.org/bios/chana_r_schoenberger">Chana R. Schoenberger</a>

                    

                    
                </p>
            </div><!-- /.meta -->
            

            <div class="feature-summary">
                
                    <p>Newly hired managers do better when integrated slowly into firm operations.</p>
                    
                    
                    
                    
                    
                    
                    
                    
                    
                
            </div>
        </div><!-- /.article-entry -->
        
    </div><!-- /.article-body -->
</article><!-- /.featured-article -->

        
        
    
        

        <article class="article article-inline">
    <div class="article-body">
        
        <div class="article-image">
            <a href="https://ssir.org/articles/entry/practicing-healthy-organizational-rituals">

                
                    
                        <img src="https://ssir.org/images/jcogs_img/cache/organizational-health-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95.webp" width="737" height="578" alt="" srcset="https://ssir.org/images/jcogs_img/cache/organizational-health-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_187w.webp 187w, https://ssir.org/images/jcogs_img/cache/organizational-health-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_290w.webp 290w, https://ssir.org/images/jcogs_img/cache/organizational-health-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_420w.webp 420w, https://ssir.org/images/jcogs_img/cache/organizational-health-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_543w.webp 543w, https://ssir.org/images/jcogs_img/cache/organizational-health-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/organizational-health-737x578_-_abcdef_-_bc5542e46553e3e88a98cdc0c95d5720022ffa95.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), (max-width: 996px) calc((100vw - 40px) * .1953), (max-width: 1994px) 187px, (max-width: 2991px) 420px, 737px" />
                    
                
            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->

        <div class="article-entry">
            
            <h5>




    
        
            
                
                    <a href="https://ssir.org/topics/category/nonprofits-ngos">Nonprofits & NGOs</a>
                

                

                
            
        

        
    
</h5>

            <h4>
                <a href="https://ssir.org/articles/entry/practicing-healthy-organizational-rituals">The Four Principles of a Breathing Organization</a>
                <script defer>fetchAndInjectHTML('/defer-load/edit-this/19965', document.currentScript);</script>
            </h4>

            
            <div class="meta">
                <p>
                    
                        

	By <a class="author" href="https://ssir.org/bios/james-lopata">James Lopata</a>

                    

                    
                        <span>
                            <a title="Comments" href="https://ssir.org/articles/entry/practicing-healthy-organizational-rituals#comments">2 <i class="ico ico-comment" aria-hidden="true"></i></a>
                        </span>
                    
                </p>
            </div><!-- /.meta -->
            

            <div class="feature-summary">
                
                    
                    
                    <p>From Model Ts to tea, organizations devoted to human flourishing need to build the human architecture for their people to breathe.</p>
                    
                    
                    
                    
                    
                    
                    
                
            </div>
        </div><!-- /.article-entry -->
        
    </div><!-- /.article-body -->
</article><!-- /.featured-article -->

        
        </div><!-- /.section-body -->
        </section><!-- /.section-articles-inline -->
        
        
    

    <div id='div-gpt-ad-1646354848641-0' style='min-width: 300px; min-height: 60px; margin-bottom: 20px; text-align: center;'>
<script>
googletag.cmd.push(function() {googletag.display('div-gpt-ad-1646354848641-0');});
</script>
</div>

    
    <section class="section section-default section-homepage">
        <header class="section-head">
            <h3>Latest</h3>
        </header><!-- /.section-head -->

        <div class="section-body">
                <article class="article article-block article-microsites list-width">	
    
    
    
        <div class="article-image">
            <a href="https://ssir.org/renewing-public-education-series">
                
    <img src="https://ssir.org/images/jcogs_img/cache/education-purpose-series-1000x750_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp" width="727" height="545" loading="lazy" alt="Renewing Public Education&#8217;s Purpose - Thumbnail" srcset="https://ssir.org/images/jcogs_img/cache/education-purpose-series-1000x750_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_102w.webp 102w, https://ssir.org/images/jcogs_img/cache/education-purpose-series-1000x750_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_300w.webp 300w, https://ssir.org/images/jcogs_img/cache/education-purpose-series-1000x750_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_480w.webp 480w, https://ssir.org/images/jcogs_img/cache/education-purpose-series-1000x750_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/education-purpose-series-1000x750_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), 102px" />

            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->
    
    <div class="article-body">
        <div class="article-entry">
            
            
            <h5 class="sponsored">
                <span>sponsored</span>
            </h5>
            

            <h4>
                <a href="https://ssir.org/renewing-public-education-series">Renewing Public Education&#8217;s Purpose</a>
            </h4>

            

            
            
            
            
            
            
            
            
            
            
            <p>The William and Flora Hewlett Foundation and the Stuart Foundation are pleased to co-sponsor this series of diverse essays on the purpose of public education. The authors write from different vantage points, but each takes seriously a core question: In a time of widespread change, what is public education for, and how can it evolve to meet its promise?</p>


            

        </div><!-- /.article-entry -->
    </div><!-- /.article-body -->
</article><!-- /.article article-block -->    <article class="article article-block article-blog list-width">	
    
    
    
        <div class="article-image">
            <a href="https://ssir.org/articles/entry/nonprofit-funding-enterprise-capital">
                
    <img src="https://ssir.org/images/jcogs_img/cache/nonprofits-enterprise-capital-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp" width="727" height="727" loading="lazy" alt="" srcset="https://ssir.org/images/jcogs_img/cache/nonprofits-enterprise-capital-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_102w.webp 102w, https://ssir.org/images/jcogs_img/cache/nonprofits-enterprise-capital-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_300w.webp 300w, https://ssir.org/images/jcogs_img/cache/nonprofits-enterprise-capital-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_480w.webp 480w, https://ssir.org/images/jcogs_img/cache/nonprofits-enterprise-capital-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/nonprofits-enterprise-capital-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), 102px" />

            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->
    
    <div class="article-body">
        <div class="article-entry">
            
            
                <h5>
                    
                    
                    

                    
                        
                            
                                
                                    
                                        <a href="https://ssir.org/topics/category/nonprofits-ngos">Nonprofits & NGOs</a>
                                    
                                
                            
                        
                    
                </h5>
            

            <h4>
                <a href="https://ssir.org/articles/entry/nonprofit-funding-enterprise-capital">Nonprofits Need Enterprise Capital to Succeed</a>
            </h4>

            
            <p class="meta">
                
                
                     By <a class="author" href="https://ssir.org/bios/andrea-levere">Andrea Levere</a>
                

                
                    <span>
                        <a title="Comments" href="https://ssir.org/articles/entry/nonprofit-funding-enterprise-capital#comments">1 <i class="ico ico-comment" aria-hidden="true"></i></a>
                    </span>
                
            </p><!-- /.meta -->
            

            
            
            <p><p>Many nonprofits face a mismatch of their budget and their balance sheet. Funders can help build stronger financial foundations.</p></p>
            
            
            
            
            
            
            
            


            

        </div><!-- /.article-entry -->
    </div><!-- /.article-body -->
</article><!-- /.article article-block -->    <article class="article article-block article-blog list-width">	
    
    
    
        <div class="article-image">
            <a href="https://ssir.org/articles/entry/due-diligence-deeper-partnerships">
                
    <img src="https://ssir.org/images/jcogs_img/cache/due-diligence-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp" width="727" height="727" loading="lazy" alt="" srcset="https://ssir.org/images/jcogs_img/cache/due-diligence-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_102w.webp 102w, https://ssir.org/images/jcogs_img/cache/due-diligence-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_300w.webp 300w, https://ssir.org/images/jcogs_img/cache/due-diligence-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_480w.webp 480w, https://ssir.org/images/jcogs_img/cache/due-diligence-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/due-diligence-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), 102px" />

            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->
    
    <div class="article-body">
        <div class="article-entry">
            
            
                <h5>
                    
                    
                    

                    
                        
                            
                                
                                    
                                        <a href="https://ssir.org/topics/category/philanthropy-funding">Philanthropy & Funding</a>
                                    
                                
                            
                        
                    
                </h5>
            

            <h4>
                <a href="https://ssir.org/articles/entry/due-diligence-deeper-partnerships">Due Diligence as a Catalyst for Growth</a>
            </h4>

            
            <p class="meta">
                
                
                     By <a class="author" href="https://ssir.org/bios/geraldine-moreno">Geraldine Moreno</a>
                

                
                    <span>
                        <a title="Comments" href="https://ssir.org/articles/entry/due-diligence-deeper-partnerships#comments">2 <i class="ico ico-comment" aria-hidden="true"></i></a>
                    </span>
                
            </p><!-- /.meta -->
            

            
            
            <p><p>Why philanthropy should think of due diligence not as a vetting exercise, but as an opportunity to build deeper partnerships that lead to more sustainable impact.</p></p>
            
            
            
            
            
            
            
            


            

        </div><!-- /.article-entry -->
    </div><!-- /.article-body -->
</article><!-- /.article article-block -->    <article class="article article-block article-blog list-width">	
    
    
    
        <div class="article-image">
            <a href="https://ssir.org/articles/entry/scale-really-matters">
                
    <img src="https://ssir.org/images/jcogs_img/cache/scale-really-matters-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp" width="727" height="727" loading="lazy" alt="" srcset="https://ssir.org/images/jcogs_img/cache/scale-really-matters-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_102w.webp 102w, https://ssir.org/images/jcogs_img/cache/scale-really-matters-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_300w.webp 300w, https://ssir.org/images/jcogs_img/cache/scale-really-matters-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_480w.webp 480w, https://ssir.org/images/jcogs_img/cache/scale-really-matters-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/scale-really-matters-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), 102px" />

            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->
    
    <div class="article-body">
        <div class="article-entry">
            
            
                <h5>
                    
                    
                    

                    
                        
                            
                                
                                    
                                        <a href="https://ssir.org/topics/category/scaling">Scaling</a>
                                    
                                
                            
                        
                    
                </h5>
            

            <h4>
                <a href="https://ssir.org/articles/entry/scale-really-matters">Scale Really Matters</a>
            </h4>

            
            <p class="meta">
                
                
                     By <a class="author" href="https://ssir.org/bios/kevin_starr">Kevin Starr</a>
                

                
                    <span>
                        <a title="Comments" href="https://ssir.org/articles/entry/scale-really-matters#comments">9 <i class="ico ico-comment" aria-hidden="true"></i></a>
                    </span>
                
            </p><!-- /.meta -->
            

            
            
            <p><p>The problems are big, the time is short, and the resources are limited.</p></p>
            
            
            
            
            
            
            
            


            

        </div><!-- /.article-entry -->
    </div><!-- /.article-body -->
</article><!-- /.article article-block -->    <article class="article article-block article-blog list-width">	
    
    
    
        <div class="article-image">
            <a href="https://ssir.org/articles/entry/nonprofit-fundraising-sanctions-ecuador">
                
    <img src="https://ssir.org/images/jcogs_img/cache/compliance-silence-new-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp" width="727" height="727" loading="lazy" alt="" srcset="https://ssir.org/images/jcogs_img/cache/compliance-silence-new-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_102w.webp 102w, https://ssir.org/images/jcogs_img/cache/compliance-silence-new-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_300w.webp 300w, https://ssir.org/images/jcogs_img/cache/compliance-silence-new-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_480w.webp 480w, https://ssir.org/images/jcogs_img/cache/compliance-silence-new-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/compliance-silence-new-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), 102px" />

            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->
    
    <div class="article-body">
        <div class="article-entry">
            
            
                <h5>
                    
                    
                    

                    
                        
                            
                                
                                    
                                        <a href="https://ssir.org/topics/category/nonprofits-ngos">Nonprofits & NGOs</a>
                                    
                                
                            
                        
                    
                </h5>
            

            <h4>
                <a href="https://ssir.org/articles/entry/nonprofit-fundraising-sanctions-ecuador">When Compliance Aims to Silence</a>
            </h4>

            
            <p class="meta">
                
                
                     By <a class="author" href="https://ssir.org/bios/tania-davila-paredes">Tania Dávila Paredes</a>
                

                
            </p><!-- /.meta -->
            

            
            
            <p><p>A look at the motivations behind and impact of a new law limiting nonprofit fundraising in Ecuador, and how civil society organizations are coming together to reclaim their agency.</p></p>
            
            
            
            
            
            
            
            


            

        </div><!-- /.article-entry -->
    </div><!-- /.article-body -->
</article><!-- /.article article-block -->

        </div><!-- /.section-body -->

        <div class="section-actions">
            <a href="https://ssir.org/articles" class="link-more">
                Read More »
            </a>
        </div><!-- /.section-actions -->
    </section><!-- /.section section-default -->





<section class="section-past-issues">
    <header class="section-head">
        <h3>browse past issues <a href="https://ssir.org/issue">all issues <span class="mobile-only tablet-only">&raquo;</span></a></h3>
    </header><!-- /.section-head -->

    <div class="section-body">
        <ul>
            
            <li>
                <a href="https://ssir.org/issue/spring-2026">
                    
                    
                        
                            <img src="https://ssir.org/images/jcogs_img/cache/spring-2026-770x1009_-_abcdef_-_9f1768fe8c9dcf32d6098823b568c67a70a27d86.webp" alt="" width="130" height="170" loading="lazy" />
                        
                    
                    
                    <span>Spring 2026</span>
                </a>
            </li>
            
            <li>
                <a href="https://ssir.org/issue/winter-2026">
                    
                    
                        
                            <img src="https://ssir.org/images/jcogs_img/cache/winter-2026-cover-770x1009_-_abcdef_-_9f1768fe8c9dcf32d6098823b568c67a70a27d86.webp" alt="" width="130" height="170" loading="lazy" />
                        
                    
                    
                    <span>Winter 2026</span>
                </a>
            </li>
            
            <li>
                <a href="https://ssir.org/issue/fall-2025">
                    
                    
                        
                            <img src="https://ssir.org/images/jcogs_img/cache/fall-2025-cover-770x1009_-_abcdef_-_9f1768fe8c9dcf32d6098823b568c67a70a27d86.webp" alt="" width="130" height="170" loading="lazy" />
                        
                    
                    
                    <span>Fall 2025</span>
                </a>
            </li>
            
            <li>
                <a href="https://ssir.org/issue/summer-2025">
                    
                    
                        
                            <img src="https://ssir.org/images/jcogs_img/cache/summer-2025-cover-770x1009_-_abcdef_-_9f1768fe8c9dcf32d6098823b568c67a70a27d86.webp" alt="" width="130" height="170" loading="lazy" />
                        
                    
                    
                    <span>Summer 2025</span>
                </a>
            </li>
            
        </ul>
    </div><!-- /.section-body -->
</section><!-- /.section-past-issues -->


<section class="section section-default section-bordered section-homepage">
    <header class="section-head">
        <h3>Most Commented</h3>
    </header><!-- /.section-head -->

    <div class="section-body">
        
        <article class="article article-block article-blog list-width">	
    
    
    
        <div class="article-image">
            <a href="https://ssir.org/articles/entry/no-such-thing-as-impact-investing">
                
    <img src="https://ssir.org/images/jcogs_img/cache/impact-investing-intentions-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp" width="727" height="727" loading="lazy" alt="" srcset="https://ssir.org/images/jcogs_img/cache/impact-investing-intentions-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_102w.webp 102w, https://ssir.org/images/jcogs_img/cache/impact-investing-intentions-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_300w.webp 300w, https://ssir.org/images/jcogs_img/cache/impact-investing-intentions-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_480w.webp 480w, https://ssir.org/images/jcogs_img/cache/impact-investing-intentions-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/impact-investing-intentions-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), 102px" />

            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->
    
    <div class="article-body">
        <div class="article-entry">
            
            
                <h5>
                    
                    
                    

                    
                        
                            
                                
                                    
                                        <a href="https://ssir.org/topics/category/impact-investing">Impact Investing</a>
                                    
                                
                            
                        
                    
                </h5>
            

            <h4>
                <a href="https://ssir.org/articles/entry/no-such-thing-as-impact-investing">There Is No Such Thing as Impact Investing</a>
            </h4>

            
            <p class="meta">
                
                
                     By <a class="author" href="https://ssir.org/bios/kevin_starr">Kevin Starr</a>
                

                
                    <span>
                        <a title="Comments" href="https://ssir.org/articles/entry/no-such-thing-as-impact-investing#comments">12 <i class="ico ico-comment" aria-hidden="true"></i></a>
                    </span>
                
            </p><!-- /.meta -->
            

            
            
            <p><p>There is philanthropic investing, and there is commercial investing, and there is nothing in between.</p></p>
            
            
            
            
            
            
            
            


            

        </div><!-- /.article-entry -->
    </div><!-- /.article-body -->
</article><!-- /.article article-block -->
        
        <article class="article article-block article-blog list-width">	
    
    
    
        <div class="article-image">
            <a href="https://ssir.org/articles/entry/scale-really-matters">
                
    <img src="https://ssir.org/images/jcogs_img/cache/scale-really-matters-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp" width="727" height="727" loading="lazy" alt="" srcset="https://ssir.org/images/jcogs_img/cache/scale-really-matters-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_102w.webp 102w, https://ssir.org/images/jcogs_img/cache/scale-really-matters-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_300w.webp 300w, https://ssir.org/images/jcogs_img/cache/scale-really-matters-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_480w.webp 480w, https://ssir.org/images/jcogs_img/cache/scale-really-matters-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/scale-really-matters-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), 102px" />

            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->
    
    <div class="article-body">
        <div class="article-entry">
            
            
                <h5>
                    
                    
                    

                    
                        
                            
                                
                                    
                                        <a href="https://ssir.org/topics/category/scaling">Scaling</a>
                                    
                                
                            
                        
                    
                </h5>
            

            <h4>
                <a href="https://ssir.org/articles/entry/scale-really-matters">Scale Really Matters</a>
            </h4>

            
            <p class="meta">
                
                
                     By <a class="author" href="https://ssir.org/bios/kevin_starr">Kevin Starr</a>
                

                
                    <span>
                        <a title="Comments" href="https://ssir.org/articles/entry/scale-really-matters#comments">9 <i class="ico ico-comment" aria-hidden="true"></i></a>
                    </span>
                
            </p><!-- /.meta -->
            

            
            
            <p><p>The problems are big, the time is short, and the resources are limited.</p></p>
            
            
            
            
            
            
            
            


            

        </div><!-- /.article-entry -->
    </div><!-- /.article-body -->
</article><!-- /.article article-block -->
        
        <article class="article article-block article-articles list-width">	
    
    
    
        <div class="article-image">
            <a href="https://ssir.org/articles/entry/nonprofits-innovation-strategies">
                
    <img src="https://ssir.org/images/jcogs_img/cache/winter-2026-viewpoint-kemp-nonprofit-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp" width="727" height="727" loading="lazy" alt="" srcset="https://ssir.org/images/jcogs_img/cache/winter-2026-viewpoint-kemp-nonprofit-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_102w.webp 102w, https://ssir.org/images/jcogs_img/cache/winter-2026-viewpoint-kemp-nonprofit-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_300w.webp 300w, https://ssir.org/images/jcogs_img/cache/winter-2026-viewpoint-kemp-nonprofit-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_480w.webp 480w, https://ssir.org/images/jcogs_img/cache/winter-2026-viewpoint-kemp-nonprofit-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/winter-2026-viewpoint-kemp-nonprofit-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), 102px" />

            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->
    
    <div class="article-body">
        <div class="article-entry">
            
            
                <h5>
                    
                    
                    

                    
                        
                            
                                
                                    
                                        <a href="https://ssir.org/topics/category/nonprofits-ngos">Nonprofits & NGOs</a>
                                    
                                
                            
                        
                    
                </h5>
            

            <h4>
                <a href="https://ssir.org/articles/entry/nonprofits-innovation-strategies">How Will Your Nonprofit Survive?</a>
            </h4>

            
            <p class="meta">
                
                
                     By <a class="author" href="https://ssir.org/bios/bo-kemp">Bo Kemp</a>
                

                
                    <span>
                        <a title="Comments" href="https://ssir.org/articles/entry/nonprofits-innovation-strategies#comments">9 <i class="ico ico-comment" aria-hidden="true"></i></a>
                    </span>
                
            </p><!-- /.meta -->
            

            <p><p>In the face of current funding uncertainty, US nonprofits must innovate to sustain their missions.</p></p>
            
            
            
            
            
            
            
            
            
            


            

        </div><!-- /.article-entry -->
    </div><!-- /.article-body -->
</article><!-- /.article article-block -->
        
        <article class="article article-block article-articles list-width">	
    
    
    
        <div class="article-image">
            <a href="https://ssir.org/articles/entry/era-of-relational-intelligence">
                
    <img src="https://ssir.org/images/jcogs_img/cache/spring-2026-feature-ai-intel-rainbow-thumb-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp" width="727" height="727" loading="lazy" alt="" srcset="https://ssir.org/images/jcogs_img/cache/spring-2026-feature-ai-intel-rainbow-thumb-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_102w.webp 102w, https://ssir.org/images/jcogs_img/cache/spring-2026-feature-ai-intel-rainbow-thumb-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_300w.webp 300w, https://ssir.org/images/jcogs_img/cache/spring-2026-feature-ai-intel-rainbow-thumb-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_480w.webp 480w, https://ssir.org/images/jcogs_img/cache/spring-2026-feature-ai-intel-rainbow-thumb-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/spring-2026-feature-ai-intel-rainbow-thumb-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), 102px" />

            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->
    
    <div class="article-body">
        <div class="article-entry">
            
            
                <h5>
                    
                    
                    

                    
                        
                            
                                
                                    
                                        <a href="https://ssir.org/topics/category/education">Education</a>
                                    
                                
                            
                        
                    
                </h5>
            

            <h4>
                <a href="https://ssir.org/articles/entry/era-of-relational-intelligence">Welcome to the Era of Relational Intelligence</a>
            </h4>

            
            <p class="meta">
                
                
                     By <a class="author" href="https://ssir.org/bios/isabelle-c-hau">Isabelle C. Hau</a>
                

                
                    <span>
                        <a title="Comments" href="https://ssir.org/articles/entry/era-of-relational-intelligence#comments">8 <i class="ico ico-comment" aria-hidden="true"></i></a>
                    </span>
                
            </p><!-- /.meta -->
            

            <p><p>As AI begins to transform education, work, and social life, we need to focus
on developing and expanding capacities essential for human flourishing.</p></p>
            
            
            
            
            
            
            
            
            
            


            

        </div><!-- /.article-entry -->
    </div><!-- /.article-body -->
</article><!-- /.article article-block -->
        
        <article class="article article-block article-blog list-width">	
    
    
    
        <div class="article-image">
            <a href="https://ssir.org/articles/entry/ai-recovers-usaid-lessons">
                
    <img src="https://ssir.org/images/jcogs_img/cache/preserving-usaid-archive-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp" width="727" height="727" loading="lazy" alt="" srcset="https://ssir.org/images/jcogs_img/cache/preserving-usaid-archive-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_102w.webp 102w, https://ssir.org/images/jcogs_img/cache/preserving-usaid-archive-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_300w.webp 300w, https://ssir.org/images/jcogs_img/cache/preserving-usaid-archive-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_480w.webp 480w, https://ssir.org/images/jcogs_img/cache/preserving-usaid-archive-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df_727w.webp 727w, https://ssir.org/images/jcogs_img/cache/preserving-usaid-archive-737x737_-_abcdef_-_8c74bbea646e433c7f113091ccacaaa86c00f9df.webp 728w" sizes="(max-width: 767px) calc(100vw - 40px), 102px" />

            </a>
              
  	
	
  

  
  
  
        </div><!-- /.article-image -->
    
    <div class="article-body">
        <div class="article-entry">
            
            
                <h5>
                    
                    
                    

                    
                        
                            
                                
                                    
                                        <a href="https://ssir.org/topics/category/economic-development">Economic Development</a>
                                    
                                
                            
                        
                    
                </h5>
            

            <h4>
                <a href="https://ssir.org/articles/entry/ai-recovers-usaid-lessons">When USAID Shut Down, Its Lessons Nearly Vanished. AI Helped Recover Them</a>
            </h4>

            
            <p class="meta">
                
                
                     By <a class="author" href="https://ssir.org/bios/lindsey-moore">Lindsey Moore</a>
                

                
                    <span>
                        <a title="Comments" href="https://ssir.org/articles/entry/ai-recovers-usaid-lessons#comments">7 <i class="ico ico-comment" aria-hidden="true"></i></a>
                    </span>
                
            </p><!-- /.meta -->
            

            
            
            <p><p>A final sweep of 60 years of evidence reveals durable truths about how development succeeds and fails.</p></p>
            
            
            
            
            
            
            
            


            

        </div><!-- /.article-entry -->
    </div><!-- /.article-body -->
</article><!-- /.article article-block -->
        
    </div><!-- /.section-body -->

    <div class="section-actions">
        
        <a href="https://ssir.org/articles/most_commented" class="link-more">
            Read More »
        </a>
    </div><!-- /.section-actions -->
</section><!-- /.section section-default -->


<div id='div-gpt-ad-1596831006998-0' style='width: 612px; height: 105px;'>
    <script>
        googletag.cmd.push(function() {googletag.display('div-gpt-ad-1596831006998-0');});
    </script>
</div>


                </section>

                <aside class="sidebar">
					
                    
                    <!-- Default sidebar -->
                    <ul class="widgets">
                        
<li class="widget widget-socials">
    <div class="widget-body">
        <a href="#">
            <img src="/css/images/logo-red.png" alt="" width="206" height="39">
        </a>

        <div class="socials">
            <ul>
                 <li>
                    <a href="https://www.linkedin.com/company/stanford-social-innovation-review" aria-label="SSIR Linkedin">
                        <i class="ico-linkedin-large"></i>
                    </a>
                </li>
                <li>
                    <a href="https://www.instagram.com/stanford.ssir/" aria-label="SSIR Instagram">
                        <img src="/images/homepage/ico-instagram-large.png" alt="SSIR Instagram" width="80px" style="max-width:80px;">
                    </a>
                </li>
                <li>
                    <a href="https://bsky.app/profile/ssir.org" aria-label="SSIR Bluesky">
                        <img src="/images/homepage/ico-bluesky-large.png" alt="SSIR Bluesky" width="80px" style="max-width:80px;">
                    </a>
                </li>
            </ul>
        </div><!-- /.socials -->
    </div><!-- /.widget-body -->
</li>
                        <li class="widget widget-banner">
    <div class="widget-body desktop-ad-container" data-ad-key="widget_ads_top">
        
        <div id='div-gpt-ad-1596666215664-0' style='width: 300px; height: 250px;'>
            <script>
              	googletag.cmd.push(function() {googletag.display('div-gpt-ad-1596666215664-0');});
            </script>
        </div>
    </div><!-- /.widget-body -->
</li>
                        <li class="widget widget-subscribe">
    <a href="https://subscriber.ssir.org/">
        <img class="js-no-account-only widget-subscribe__img" src="/assets/img/sidebar-sign-up-free-account.png" alt="Free Weekly eNewsletter" loading="lazy" style="display: none;" />
        <img class="js-free-only widget-subscribe__img" src="/assets/img/sidebar-sign-up-subscribe.png" alt="Free Weekly eNewsletter" loading="lazy" style="display: none;" />
    </a>
</li>
                        <li class="widget">
    <a href="https://ssir.org/donate"><img src="/css/images/support_ssir_300_250.jpeg" alt="Support SSIR"/></a>
</li>
                        <li class="widget widget-banner">
    <div class="widget-body desktop-ad-container" data-ad-key="widget_ads_bottom">
		
        <div id='div-gpt-ad-1596665496550-0' style='width: 300px; height: 250px;'>
            <script>
              	googletag.cmd.push(function() {googletag.display('div-gpt-ad-1596665496550-0');});
            </script>
        </div>
    </div><!-- /.widget-body -->
</li>
                        

                        

                        

                        
                        

                        <li class="widget widget-published-by widget-homepage">
    <div class="widget-body">
        <a href="http://pacscenter.stanford.edu/" aria-label="Stanford Center on Philanthropy and Civil Society">
            <img src="/css/images/StanfordPACS_Wht_HR_resized.png" alt="Stanford Center on Philanthropy and Civil Society Logo" class="pacs-logo" width="269" height="83" />
        </a>
        <br />
        <em>SSIR</em> is published by the<br />
        <a href="http://pacscenter.stanford.edu/">Stanford Center on Philanthropy and Civil<br />
        Society</a>, at <a href="http://www.stanford.edu/">Stanford University</a>.
    </div><!-- published-by -->
</li>

                                                

                    </ul>
                    
                </aside>
            </div>

                </div>
            </div>

            <footer class="footer">
        <div class="footer-top">
            <div class="shell">
                <div class="footer-top-inner">
                    <nav class="footer-nav">
                        <ul>
                            	<li>
		<a href="https://ssir.org/about/overview">About <em>SSIR</em></a>
	</li>
	<li>
		<a href="http://pacscenter.stanford.edu/">Stanford PACS</a>
	</li>
	<li>
		<a href="https://ssir.org/about/contact">Contact Us</a>
	</li>
	<li>
		<a href="https://ssir.org/about/submission-guidelines">Submissions</a>
	</li>
	<li>
		<a href="https://ssir.org/advertising">Advertise</a>
	</li>
	<li>
		<a href="https://ssir.org/help">Help</a>
	</li>
	<li>
		<a href="https://ssir.org/reprints-permissions">Reprints</a>
	</li>
	<li>
		<a href="http://www.stanford.edu/site/terms.html">Terms of Use</a>
	</li>
	<li>
		<a href="https://ssir.org/privacy-policy">Privacy</a>
	</li>
		<li>
		<a href="https://www.stanford.edu/site/accessibility/">Accessibility</a>
	</li>
	<li>
		<a href="https://ssir.org/site-map">Site Map</a>
	</li>

                        </ul>
                    </nav><!-- /.footer-nav -->
                </div><!-- /.footer-top-inner -->
            </div><!-- /.shell -->
        </div><!-- /.footer-top -->
        <div class="footer-bottom">
            <div class="shell">
                <div class="footer-bottom-inner">
                    <p class="copyright">Copyright &copy; 2026 Stanford University. <span>Designed by <a href="#">Arsenal</a>, developed by <a href="http://www.hopstudios.com/" title="Web design and ExpressionEngine developers">Hop Studios</a></span></p><!-- /.copyright -->
                    <button type="button" class="manage-cookie" style="background: none; width: 165px; height: 30px; border-radius: 3px; border: 1px solid #900; font-family: Roboto, Arial; font-size: 10px; color: #900; font-weight: 600; box-sizing: border-box; cursor: pointer; padding: 0; outline: none; margin-top: 8px;" onclick="displayPreferenceModal()">Manage Cookie Preferences</button>
                </div><!-- /.footer-bottom-inner -->
            </div><!-- /.shell -->
        </div><!-- /.footer-bottom -->
    </footer><!-- /.footer -->

        </div><!-- /.wrapper -->

        <script>
            let articleStatus, articleSponsored;

            
            
        </script>

        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.1/jquery.min.js"></script>
<script src="https://ssir.org/minify/f9a628747dcb0d23e808a6deab83b6cf5bc66c02.1528146029.js"></script>
<script src="https://ssir.org/site/functions?v=2026-02-20-6" defer></script>

<script src="https://subscriber.ssir.org/c/assets/pigeon.js" onload="console.log('pigeon.js loaded')"
    onerror="console.error('pigeon.js failed to load')"></script>
<script type="text/javascript">
    const Pigeon = new PigeonClass({
        subdomain: 'subscriber.ssir.org',
        fingerprint: true
    });
    // Mobile-only sidebar offset settings.
    // We keep the sidebar below the paywall promo panel with an extra 20px gap.
    const MOBILE_SIDEBAR_MEDIA_QUERY = '(max-width: 767px)';
    const MOBILE_SIDEBAR_TOP_GAP = 20;

    // Positions `.sidebar` below the paywall promo on mobile.
    // We use promo document top + promo outer height (+ 20px), then convert
    // that absolute target into a relative `top` shift for `.sidebar`.
    const updateMobileSidebarAndFooterTopOffset = () => {
        const sidebar = document.querySelector('.sidebar');
        const footer = document.querySelector('.footer');
        const promotionContent = document.querySelector('.pigeon-context-promotion-content');
        const protectedOverlay = document.querySelector('.protected-overlay');
        if (!sidebar || !footer || !promotionContent || !protectedOverlay) return;

        if (!window.matchMedia(MOBILE_SIDEBAR_MEDIA_QUERY).matches) {
            sidebar.style.position = '';
            sidebar.style.top = '';
            footer.style.position = '';
            footer.style.marginTop = '';
            return;
        }

        sidebar.style.position = 'relative';
        sidebar.style.top = '';

        let mobileOffsetTop = Math.max(promotionContent.getBoundingClientRect().bottom - (protectedOverlay.getBoundingClientRect().top + protectedOverlay.scrollHeight), 0);

        const sidebarOffset = mobileOffsetTop + 20;
        sidebar.style.top = `${sidebarOffset}px`;

        footer.style.marginTop = '';
        const footerOffset = mobileOffsetTop + 40;
        footer.style.marginTop = `${footerOffset}px`;
    };

    const debounce = (fn, delay) => {
        let timeoutId;
        return function () {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(fn, delay);
        };
    };

    // Reserve vertical space for absolutely positioned paywall promo content
    // by adding promo height to overlay height in the container min-height.
    const updatePromotionContainerSpacing = () => {
        const isMobileView = window.matchMedia(MOBILE_SIDEBAR_MEDIA_QUERY).matches;
        document.querySelectorAll('.pigeon-context-promotion').forEach((container) => {
            const protectedOverlay = container.querySelector('.protected-overlay');
            const promotionContent = container.querySelector('.pigeon-context-promotion-content');
            if (!protectedOverlay || !promotionContent) return;

            if (isMobileView) {
                container.style.minHeight = '';
                return;
            }

            const containerDisplay = getComputedStyle(container).display;
            if (containerDisplay === 'none') {
                container.style.minHeight = '';
                return;
            }

            const promoStyles = getComputedStyle(promotionContent);
            const promoTopMargin = parseFloat(promoStyles.marginTop || '0') || 0;
            const promoBottomMargin = parseFloat(promoStyles.marginBottom || '0') || 0;
            const overlayHeight = protectedOverlay.offsetHeight || protectedOverlay.scrollHeight || 0;
            const promoHeight = promotionContent.offsetHeight + promoTopMargin + promoBottomMargin;
            const requiredHeight = Math.max(overlayHeight + promoHeight, 0);

            container.style.minHeight = `${Math.ceil(requiredHeight)}px`;
        });
    };
    const recalculatePigeonLayout = () => {
        updatePromotionContainerSpacing();
        updateMobileSidebarAndFooterTopOffset();
    };
    const recalculatePigeonLayoutDebounced = debounce(recalculatePigeonLayout, 100);

    // Run one shared layout pass on initial load and viewport size changes.
    window.addEventListener('load', recalculatePigeonLayout);
    window.addEventListener('resize', recalculatePigeonLayoutDebounced);
    setTimeout(recalculatePigeonLayout, 0);

    const free = (false || articleStatus == 'Paid Access Open' || articleSponsored == 'yes');

    Pigeon.paywall({
        redirect: false,
        free: free,
        ignoreBodyHide: true,
        success: (response) => {}});

    const PIGEON_BASE = 'https://subscriber.ssir.org';

    Promise.resolve(Pigeon.paywallPromise).then((r) => {
        const { allowed, profile, user_status, restricted, meter_limit, meter } = r;
        const limit = Number(meter_limit || 0);
        const used = Number(meter || 0);
        const remaining = Math.max(0, limit - used);
        const loggedIn = user_status === 1 || (profile && profile.id);
        const isMeteredPage = restricted === 0 && !free;
        const plans = profile && profile.plans;
        const hasPlan = Array.isArray(plans) ? plans.length > 0 : Boolean(plans);
        const isFreeCategory = Boolean(profile) && (
            plans === null ||
            (Array.isArray(plans) && plans.length === 0)
        );

        const moveMobileAdsToSidebarWidgets = () => {
            const sidebarWidgets = document.querySelector('.sidebar .widgets');
            if (!sidebarWidgets) return;
            document.querySelectorAll('.mobile-ad-container').forEach((adContainer) => {sidebarWidgets.appendChild(adContainer);});
        };
        const setVisibilityByAccess = () => {
            [
                ['.js-no-account-only', !Boolean(profile)],
                ['.js-free-only', isFreeCategory],
                ['.js-plan-only', hasPlan]
            ].forEach(([selector, isVisible]) => {
                document.querySelectorAll(selector).forEach((el) => {
                    el.style.display = isVisible ? '' : 'none';
                });
            });
        };
        setVisibilityByAccess();
        const removeRestricted = (selector) => {
            document.querySelectorAll(selector).forEach((el) => {
                if (el.dataset.loginPrompt) {
                    document.getElementById(el.dataset.loginPrompt).style.display = 'block';
                }
                el.remove();
            });
        };
        const shouldRemoveAllRestricted = !hasPlan && (!allowed || (isMeteredPage && remaining == 0));

        // Removing access required elements if the page is metered and the user has no remaining articles
        if (shouldRemoveAllRestricted) {
            removeRestricted('.js-access-required, .js-plan-required');
        } else if (!hasPlan) {removeRestricted('.js-plan-required');}

        // Desktop banner: status widget + show "pages left" only on metered pages (restricted === 0)
        const initStatusBanner = () => {
            if (typeof Pigeon.widget === 'undefined' || typeof Pigeon.widget.status !== 'function') return;
            const placeholder = document.querySelector('.widget-sticky .pigeon-widget-status');
            if (!placeholder) return;
            Pigeon.widget.status({
                customCopy: {
                    page: ['article', 'articles'],
                    meter: '%%available%% %%page%% left',
                    login: loggedIn ? 'Sign In' : 'Log in',
                    subscribe: loggedIn ? 'Subscribe' : 'Sign up'
                }
            });

            const setMeterVisibility = () => {
                const meterEl = document.querySelector('.widget-sticky .pigeon-widget-status ul li.meter');
                if (meterEl) meterEl.style.display = isMeteredPage && remaining != 0 ? '' : 'none';
                markFirstVisible();
            };
            const markFirstVisible = () => {
                const ul = document.querySelector('.widget-sticky .pigeon-widget-status ul');
                if (!ul) return;
                const items = Array.from(ul.querySelectorAll(':scope > li'));
                items.forEach(li => li.classList.remove('first-visible'));
                const firstVisible = items.find(li => li.offsetParent !== null && getComputedStyle(li).display !== 'none');
                if (firstVisible) firstVisible.classList.add('first-visible');
            };
            let attempts = 0;
            const trySetMeter = () => {
                setMeterVisibility();
                if (document.querySelector('.widget-sticky .pigeon-widget-status ul li.meter') || attempts >= 20) return;
                attempts += 1;
                setTimeout(trySetMeter, 50);
            };
            setTimeout(trySetMeter, 0);
        };
        initStatusBanner();
        const statusObs = new MutationObserver(() => {
            if (document.querySelector('.widget-sticky .pigeon-widget-status')) {
                initStatusBanner();
                statusObs.disconnect();
            }
        });
        statusObs.observe(document.body, { childList: true, subtree: true });

        // Prompt widget (slider) copy
        if (free) return;

        if (!Pigeon.widget || typeof Pigeon.widget.prompt !== 'function') {
            console.warn('Pigeon.widget.prompt is not available', Pigeon.widget);
            return;
        }
        const meterTitle = document.querySelector('.pigeon-meter-title');
        const meterMessage = document.querySelector('.pigeon-meter-message');
        const upgrade = document.querySelector('.pigeon-upgrade');
        const subscribe = document.querySelector('.pigeon-subscribe');
        const signIn = document.querySelector('.pigeon-sign-in');

        if (meterTitle && meterMessage && upgrade && subscribe && signIn) {
            upgrade.style.display = 'none';
            subscribe.style.display = 'inline-block';
            signIn.style.display = 'inline-block';
            meterMessage.innerHTML = `You have ${remaining} more free article${remaining === 1 ? '' : 's'} this month. Sign up for a free account or subscribe for additional access.`;
            if (remaining >= 1 && profile && profile.plans === null) {
                meterTitle.innerHTML = 'Thank you for reading.';
                meterMessage.innerHTML = `As a free subscriber, you have ${remaining} more free article${remaining === 1 ? '' : 's'} this month. Upgrade now to a paid subscription for unlimited access and to support <em>SSIR</em>.`;
                upgrade.style.display = 'inline-block';
                subscribe.style.display = 'none';
                signIn.style.display = 'none';
            } else if (remaining === 0) {
                if (profile && profile.plans === null) {
                    meterTitle.innerHTML = 'Support SSIR.';
                    meterMessage.innerHTML = 'This is your last free article this month. Upgrade to a paid subscription for unlimited access and to help <em>SSIR</em> share insights from social innovators around the world.';
                    upgrade.style.display = 'inline-block';
                    subscribe.style.display = 'none';
                    signIn.style.display = 'none';
                } else {
                    meterTitle.innerHTML = 'This is your last free article.';
                    meterMessage.innerHTML = '<em>SSIR</em> is a nonprofit that depends on readers like you. Sign up for a free account or subscribe for additional access.';
                }
            }
        }

        if (restricted === 1 || remaining <= 0) {
            moveMobileAdsToSidebarWidgets();
            recalculatePigeonLayout();
            if (profile && profile.plans === null) {
                const promotionContent = document.querySelector('.pigeon-context-promotion-content');
                if (promotionContent) {
                    promotionContent.innerHTML = '<h3>You’ve run out of free articles.</h3><p>Thanks for being a regular <em>SSIR</em> reader. We are a nonprofit and rely on readers like you to fund our work, so please subscribe to keep reading. For as little as $6 per month, you’ll get:</p><ul><li>Unlimited access to new articles</li><li>Access to <em>SSIR</em>’s 20+ years of field-defining work</li><li>Digital PDF editions of every print issue</li></ul><p>And you’ll make sure <em>SSIR</em> continues to support social innovators around the world with new ideas, insights, and inspiration.</p><div class="pigeon-cta-buttons"><a href="https://subscriber.ssir.org" class="pigeon-btn pigeon-btn-primary">Subscribe now</a></div><p class="pigeon-cta-footer">Interested in group subscription discounts for your organization? <a href="https://ssir.org/groups">Click here for group discounts</a>. Or <a href="https://ssir.org/gift">click here to give a gift subscription</a>.</p>';
                    // Promo HTML is injected asynchronously, so run one layout recalculation.
                    recalculatePigeonLayout();
                }
            }
            return;
        }

        Pigeon.widget.prompt({});
        recalculatePigeonLayout();
    }).catch((err) => console.error('paywallPromise failed', err));

    if (typeof MutationObserver !== 'undefined') {
        const promotionObserver = new MutationObserver(recalculatePigeonLayoutDebounced);
        document.querySelectorAll('.pigeon-context-promotion').forEach((container) => {
            promotionObserver.observe(container, {
                childList: true,
                subtree: true,
                attributes: true,
                attributeFilter: ['class', 'style']
            });
        });
    }

    // .widget-sticky: scroll down = top follows scrollY (banner up); scroll up = top by pixels traveled up from anchor
    (function () {
        let maxTopOffset = 48; // fallback until we measure
        let ticking = false;

        const initialScrollY = window.scrollY || document.documentElement.scrollTop;

        let lastScrollY = initialScrollY;

        // Initialize correctly based on scroll position
        let currentTop = initialScrollY > 0 ? -maxTopOffset : 0;

        function clamp(n, min, max) {
            return Math.max(min, Math.min(max, n));
        }

        function measureMaxTopOffset() {
            const banner = document.querySelector('.widget-sticky');
            if (banner && banner.offsetHeight > 0) {
                maxTopOffset = banner.offsetHeight;
            }
        }

        function updateBannerTop() {
            const banner = document.querySelector('.widget-sticky');
            if (!banner) return;

            measureMaxTopOffset();

            banner.style.position = 'fixed';
            banner.style.left = '0';
            banner.style.width = '100%';

            const scrollY = window.scrollY || document.documentElement.scrollTop;
            const deltaY = scrollY - lastScrollY;
            lastScrollY = scrollY;

            currentTop = clamp(currentTop - deltaY, -maxTopOffset, 0);

            banner.style.top = `${currentTop}px`;
            ticking = false;
        }

        function onScroll() {
            if (ticking) return;
            ticking = true;
            requestAnimationFrame(updateBannerTop);
        }

        function onResizeEnd() {
            measureMaxTopOffset();
            currentTop = clamp(currentTop, -maxTopOffset, 0);
            const banner = document.querySelector('.widget-sticky');
            if (banner) {
                banner.style.top = `${currentTop}px`;
            }
        }

        function debounce(fn, delay) {
            let timeoutId;
            return function () {
                clearTimeout(timeoutId);
                timeoutId = setTimeout(fn, delay);
            };
        }

        window.addEventListener('scroll', onScroll, { passive: true });
        window.addEventListener('resize', debounce(onResizeEnd, 150));

        // Force initial render state
        updateBannerTop();
    })();
</script>
<div class="pigeon-widget-prompt" style="display:none;">
    <div>
        <span class="pigeon-close" title="Close">
            <svg height="24" viewBox="0 0 24 24" width="24">
                <path
                    d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" />
                <path d="M0 0h24v24H0z" fill="none" />
            </svg>
        </span>
        <h2 class="pigeon-meter-title">This article is free.</h2>
    </div>
    <div>
        <div class="pigeon-meter-message">You have %%pigeonMeterAvailable%% more free articles this month. Sign up for a free account or subscribe for additional access.</div>
    </div>
    <div>
        <a class="pigeon-subscribe" href="#">Sign up</a>
        <a class="pigeon-sign-in" href="#">Log in</a>
        <a class="pigeon-upgrade" href="https://subscriber.ssir.org/" style="display: none;">Upgrade</a>
    </div>
</div>

        
        



<script>
  window.fbAsyncInit = function() {
    FB.init({
      appId      : '1566208430350173',
      xfbml      : true,
      version    : 'v2.9'
    });
    FB.AppEvents.logPageView();
  };

  (function(d, s, id){
     var js, fjs = d.getElementsByTagName(s)[0];
     if (d.getElementById(id)) {return;}
     js = d.createElement(s); js.id = id;
     js.src = "//connect.facebook.net/en_US/sdk.js";
     fjs.parentNode.insertBefore(js, fjs);
   }(document, 'script', 'facebook-jssdk'));
</script>



<script src="/js/share.js?v=2023-11-26"></script>
</main>
<script type="text/javascript">window.NREUM||(NREUM={});NREUM.info={"beacon":"bam.nr-data.net","licenseKey":"NRJS-59f45596d67da7d3e71","applicationID":"1064674268","transactionName":"YFRRZEdYWEtTBRZYDFkeZkJcFkVMUxILUkxERVJeU1ZEXG0VDVIKVl1sWVtXWU5TEgteDWhDVkZcXEEXQRIDRQpUHlpeUVxOFkIOEg==","queueTime":0,"applicationTime":2,"atts":"TBNSEg9CS0U=","errorBeacon":"bam.nr-data.net","agent":""}</script></body>
</html>
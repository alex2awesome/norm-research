<!-- URL: https://www.sciencemediacentre.org/ -->
<!-- HTTP: 200 SIZE: 83962 -->
 <!DOCTYPE html>
<html lang="en-GB">
<head>
<meta charset="UTF-8" />
<title>
Science Media Centre</title>
    

<link rel="profile" href="http://gmpg.org/xfn/11" />

<link rel="stylesheet" type="text/css" media="all" href="https://www.sciencemediacentre.org/wp-content/themes/SMC/assets/build/app.css" />
<meta name="viewport" content="width=device-width,initial-scale=1.0">


 
 <!--[if IE 7]>
   <link rel="stylesheet" type="text/css" media="screen" href="https://www.sciencemediacentre.org/wp-content/themes/SMC/css/ie7.css"  />
<![endif]-->

 <!--[if IE 8]>
   <link rel="stylesheet" type="text/css" media="screen" href="https://www.sciencemediacentre.org/wp-content/themes/SMC/css/ie8.css"  />
<![endif]-->

 <!--[if IE 9]>
   <link rel="stylesheet" type="text/css" media="screen" href="https://www.sciencemediacentre.org/wp-content/themes/SMC/css/ie9.css"  />
<![endif]-->

 <!--[if IE 10]>
   <link rel="stylesheet" type="text/css" media="screen" href="https://www.sciencemediacentre.org/wp-content/themes/SMC/css/ie9.css"  />
<![endif]-->




<!--[if lt IE 9]>
   <script>
      document.createElement('header');
      document.createElement('nav');
      document.createElement('section');
      document.createElement('article');
      document.createElement('aside');
      document.createElement('footer');
   </script>
<![endif]-->

<!-- Pulled from http://code.google.com/p/html5shiv/ -->
<!--[if lt IE 9]>
<script src="//html5shim.googlecode.com/svn/trunk/html5.js"></script>
<![endif]-->

<link rel="pingback" href="https://www.sciencemediacentre.org/xmlrpc.php" />

<!--Plugin Global Hide Admin Tool Bar Active-->

<!-- This website is patched against a big problem not solved from WordPress 3.3+ to date -->

<meta name='robots' content='max-image-preview:large' />
<link rel="alternate" type="application/rss+xml" title="Science Media Centre &raquo; Feed" href="https://www.sciencemediacentre.org/feed/" />
<link rel="alternate" type="application/rss+xml" title="Science Media Centre &raquo; Comments Feed" href="https://www.sciencemediacentre.org/comments/feed/" />
		<!-- This site uses the Google Analytics by MonsterInsights plugin v10.1.2 - Using Analytics tracking - https://www.monsterinsights.com/ -->
							<script src="//www.googletagmanager.com/gtag/js?id=G-HHR8B050JP"  data-cfasync="false" data-wpfc-render="false" type="text/javascript" async></script>
			<script data-cfasync="false" data-wpfc-render="false" type="text/javascript">
				var mi_version = '10.1.2';
				var mi_track_user = true;
				var mi_no_track_reason = '';
								var MonsterInsightsDefaultLocations = {"page_location":"https:\/\/www.sciencemediacentre.org\/"};
								if ( typeof MonsterInsightsPrivacyGuardFilter === 'function' ) {
					var MonsterInsightsLocations = (typeof MonsterInsightsExcludeQuery === 'object') ? MonsterInsightsPrivacyGuardFilter( MonsterInsightsExcludeQuery ) : MonsterInsightsPrivacyGuardFilter( MonsterInsightsDefaultLocations );
				} else {
					var MonsterInsightsLocations = (typeof MonsterInsightsExcludeQuery === 'object') ? MonsterInsightsExcludeQuery : MonsterInsightsDefaultLocations;
				}

								var disableStrs = [
										'ga-disable-G-HHR8B050JP',
									];

				/* Function to detect opted out users */
				function __gtagTrackerIsOptedOut() {
					for (var index = 0; index < disableStrs.length; index++) {
						if (document.cookie.indexOf(disableStrs[index] + '=true') > -1) {
							return true;
						}
					}

					return false;
				}

				/* Disable tracking if the opt-out cookie exists. */
				if (__gtagTrackerIsOptedOut()) {
					for (var index = 0; index < disableStrs.length; index++) {
						window[disableStrs[index]] = true;
					}
				}

				/* Opt-out function */
				function __gtagTrackerOptout() {
					for (var index = 0; index < disableStrs.length; index++) {
						document.cookie = disableStrs[index] + '=true; expires=Thu, 31 Dec 2099 23:59:59 UTC; path=/';
						window[disableStrs[index]] = true;
					}
				}

				if ('undefined' === typeof gaOptout) {
					function gaOptout() {
						__gtagTrackerOptout();
					}
				}
								window.dataLayer = window.dataLayer || [];

				window.MonsterInsightsDualTracker = {
					helpers: {},
					trackers: {},
				};
				if (mi_track_user) {
					function __gtagDataLayer() {
						dataLayer.push(arguments);
					}

					function __gtagTracker(type, name, parameters) {
						if (!parameters) {
							parameters = {};
						}

						if (parameters.send_to) {
							__gtagDataLayer.apply(null, arguments);
							return;
						}

						if (type === 'event') {
														parameters.send_to = monsterinsights_frontend.v4_id;
							var hookName = name;
							if (typeof parameters['event_category'] !== 'undefined') {
								hookName = parameters['event_category'] + ':' + name;
							}

							if (typeof MonsterInsightsDualTracker.trackers[hookName] !== 'undefined') {
								MonsterInsightsDualTracker.trackers[hookName](parameters);
							} else {
								__gtagDataLayer('event', name, parameters);
							}
							
						} else {
							__gtagDataLayer.apply(null, arguments);
						}
					}

					__gtagTracker('js', new Date());
					__gtagTracker('set', {
						'developer_id.dZGIzZG': true,
											});
					if ( MonsterInsightsLocations.page_location ) {
						__gtagTracker('set', MonsterInsightsLocations);
					}
										__gtagTracker('config', 'G-HHR8B050JP', {"forceSSL":"true"} );
										window.gtag = __gtagTracker;										(function () {
						/* https://developers.google.com/analytics/devguides/collection/analyticsjs/ */
						/* ga and __gaTracker compatibility shim. */
						var noopfn = function () {
							return null;
						};
						var newtracker = function () {
							return new Tracker();
						};
						var Tracker = function () {
							return null;
						};
						var p = Tracker.prototype;
						p.get = noopfn;
						p.set = noopfn;
						p.send = function () {
							var args = Array.prototype.slice.call(arguments);
							args.unshift('send');
							__gaTracker.apply(null, args);
						};
						var __gaTracker = function () {
							var len = arguments.length;
							if (len === 0) {
								return;
							}
							var f = arguments[len - 1];
							if (typeof f !== 'object' || f === null || typeof f.hitCallback !== 'function') {
								if ('send' === arguments[0]) {
									var hitConverted, hitObject = false, action;
									if ('event' === arguments[1]) {
										if ('undefined' !== typeof arguments[3]) {
											hitObject = {
												'eventAction': arguments[3],
												'eventCategory': arguments[2],
												'eventLabel': arguments[4],
												'value': arguments[5] ? arguments[5] : 1,
											}
										}
									}
									if ('pageview' === arguments[1]) {
										if ('undefined' !== typeof arguments[2]) {
											hitObject = {
												'eventAction': 'page_view',
												'page_path': arguments[2],
											}
										}
									}
									if (typeof arguments[2] === 'object') {
										hitObject = arguments[2];
									}
									if (typeof arguments[5] === 'object') {
										Object.assign(hitObject, arguments[5]);
									}
									if ('undefined' !== typeof arguments[1].hitType) {
										hitObject = arguments[1];
										if ('pageview' === hitObject.hitType) {
											hitObject.eventAction = 'page_view';
										}
									}
									if (hitObject) {
										action = 'timing' === arguments[1].hitType ? 'timing_complete' : hitObject.eventAction;
										hitConverted = mapArgs(hitObject);
										__gtagTracker('event', action, hitConverted);
									}
								}
								return;
							}

							function mapArgs(args) {
								var arg, hit = {};
								var gaMap = {
									'eventCategory': 'event_category',
									'eventAction': 'event_action',
									'eventLabel': 'event_label',
									'eventValue': 'event_value',
									'nonInteraction': 'non_interaction',
									'timingCategory': 'event_category',
									'timingVar': 'name',
									'timingValue': 'value',
									'timingLabel': 'event_label',
									'page': 'page_path',
									'location': 'page_location',
									'title': 'page_title',
									'referrer' : 'page_referrer',
								};
								for (arg in args) {
																		if (!(!args.hasOwnProperty(arg) || !gaMap.hasOwnProperty(arg))) {
										hit[gaMap[arg]] = args[arg];
									} else {
										hit[arg] = args[arg];
									}
								}
								return hit;
							}

							try {
								f.hitCallback();
							} catch (ex) {
							}
						};
						__gaTracker.create = newtracker;
						__gaTracker.getByName = newtracker;
						__gaTracker.getAll = function () {
							return [];
						};
						__gaTracker.remove = noopfn;
						__gaTracker.loaded = true;
						window['__gaTracker'] = __gaTracker;
					})();
									} else {
										console.log("");
					(function () {
						function __gtagTracker() {
							return null;
						}

						window['__gtagTracker'] = __gtagTracker;
						window['gtag'] = __gtagTracker;
					})();
									}
			</script>
							<!-- / Google Analytics by MonsterInsights -->
		<style id='wp-img-auto-sizes-contain-inline-css' type='text/css'>
img:is([sizes=auto i],[sizes^="auto," i]){contain-intrinsic-size:3000px 1500px}
/*# sourceURL=wp-img-auto-sizes-contain-inline-css */
</style>
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

<link rel='stylesheet' id='cookie-law-info-css' href='https://www.sciencemediacentre.org/wp-content/plugins/cookie-law-info/legacy/public/css/cookie-law-info-public.css?ver=3.4.0' type='text/css' media='all' />
<link rel='stylesheet' id='cookie-law-info-gdpr-css' href='https://www.sciencemediacentre.org/wp-content/plugins/cookie-law-info/legacy/public/css/cookie-law-info-gdpr.css?ver=3.4.0' type='text/css' media='all' />
<script type="text/javascript" src="https://www.sciencemediacentre.org/wp-includes/js/jquery/jquery.min.js?ver=3.7.1" id="jquery-core-js"></script>
<script type="text/javascript" src="https://www.sciencemediacentre.org/wp-includes/js/jquery/jquery-migrate.min.js?ver=3.4.1" id="jquery-migrate-js"></script>
<script type="text/javascript" src="https://www.sciencemediacentre.org/wp-content/plugins/google-analytics-for-wordpress/assets/js/frontend-gtag.min.js?ver=10.1.2" id="monsterinsights-frontend-script-js" async="async" data-wp-strategy="async"></script>
<script data-cfasync="false" data-wpfc-render="false" type="text/javascript" id='monsterinsights-frontend-script-js-extra'>/* <![CDATA[ */
var monsterinsights_frontend = {"js_events_tracking":"true","download_extensions":"doc,pdf,ppt,zip,xls,docx,pptx,xlsx","inbound_paths":"[]","home_url":"https:\/\/www.sciencemediacentre.org","hash_tracking":"false","v4_id":"G-HHR8B050JP"};/* ]]> */
</script>
<script type="text/javascript" id="cookie-law-info-js-extra">
/* <![CDATA[ */
var Cli_Data = {"nn_cookie_ids":[],"cookielist":[],"non_necessary_cookies":[],"ccpaEnabled":"","ccpaRegionBased":"","ccpaBarEnabled":"","strictlyEnabled":["necessary","obligatoire"],"ccpaType":"gdpr","js_blocking":"","custom_integration":"","triggerDomRefresh":"","secure_cookies":""};
var cli_cookiebar_settings = {"animate_speed_hide":"500","animate_speed_show":"500","background":"#fff","border":"#444","border_on":"","button_1_button_colour":"#2228d8","button_1_button_hover":"#1b20ad","button_1_link_colour":"#fff","button_1_as_button":"1","button_1_new_win":"","button_2_button_colour":"#333","button_2_button_hover":"#292929","button_2_link_colour":"#444","button_2_as_button":"","button_2_hidebar":"","button_3_button_colour":"#000000","button_3_button_hover":"#000000","button_3_link_colour":"#ffffff","button_3_as_button":"1","button_3_new_win":"1","button_4_button_colour":"#000","button_4_button_hover":"#000000","button_4_link_colour":"#fff","button_4_as_button":"1","button_7_button_colour":"#61a229","button_7_button_hover":"#4e8221","button_7_link_colour":"#fff","button_7_as_button":"1","button_7_new_win":"","font_family":"inherit","header_fix":"","notify_animate_hide":"1","notify_animate_show":"","notify_div_id":"#cookie-law-info-bar","notify_position_horizontal":"right","notify_position_vertical":"bottom","scroll_close":"","scroll_close_reload":"","accept_close_reload":"1","reject_close_reload":"1","showagain_tab":"","showagain_background":"#fff","showagain_border":"#000","showagain_div_id":"#cookie-law-info-again","showagain_x_position":"100px","text":"#000","show_once_yn":"","show_once":"10000","logging_on":"","as_popup":"fffffff","popup_overlay":"1","bar_heading_text":"","cookie_bar_as":"banner","popup_showagain_position":"bottom-right","widget_position":"left"};
var log_object = {"ajax_url":"https://www.sciencemediacentre.org/wp-admin/admin-ajax.php"};
//# sourceURL=cookie-law-info-js-extra
/* ]]> */
</script>
<script type="text/javascript" src="https://www.sciencemediacentre.org/wp-content/plugins/cookie-law-info/legacy/public/js/cookie-law-info-public.js?ver=3.4.0" id="cookie-law-info-js"></script>
<link rel="https://api.w.org/" href="https://www.sciencemediacentre.org/wp-json/" /><link rel="EditURI" type="application/rsd+xml" title="RSD" href="https://www.sciencemediacentre.org/xmlrpc.php?rsd" />
<meta name="generator" content="WordPress 6.9.4" />

<!-- Contact Us Form -->
<style type="text/css">
.cuf_input {display:none !important; visibility:hidden !important;}
#contactsubmit:hover, #contactsubmit:focus {
	background: #849F00 repeat-x;
	color: #FFF;
	text-decoration: none;
}

#contactsubmit {
	font-family: 'Oxygen', sans-serif;
	font-size:16px;
	color:#0c4da2;
	display: block;
	float: left;
	font-size: 16px;
	text-decoration: none;
	text-align:center;
	background: rgba(255, 255, 255, 1); 
padding-top:10px;
padding-bottom:10px;
padding-left:20px;
padding-right:20px;
-moz-border-radius: 18px;
border-radius: 18px;
 border:none;
  cursor:pointer;
}

#contactsubmit:hover {
background: rgba(17, 17, 17, 1);
-webkit-transition: all 0.2s ease-in-out;
   -moz-transition: all 0.2s ease-in-out;
   -o-transition: all 0.2s ease-in-out;
   -ms-transition: all 0.2s ease-in-out;
   transition: all 0.2s ease-in-out;
}


.cuf_field {
	-moz-box-sizing:border-box;
	-webkit-box-sizing:border-box;
	box-sizing:border-box;
	background:#fff;
	border:1px solid #A9B3BC;
	padding:8px;
	width:100%;
	margin-top:5px;
	margin-bottom:15px;
	outline:none
}
#tinyform {
clear: both;
	width:550px;
	//margin-left:auto;
	//margin-right:auto;
	/*margin-top:30px;*/
	padding:20px;
	-webkit-border-radius:5px;
	-moz-border-radius:5px;
	border-radius:5px;
	-webkit-box-shadow:0px 0px 10px 0px rgba(0,0,0,0.2);
	-moz-box-shadow:0px 0px 10px 0px rgba(0,0,0,0.2);
	//box-shadow:0px 0px 10px 0px rgba(0,0,0,0.2);
	border:4px solid #FFF;
	-webkit-transition:all 200ms linear;
	-moz-transition:all 200ms linear;
	-o-transition:all 200ms linear;
	transition:all 200ms linear;
}
.cuf_textarea {
	-moz-box-sizing:border-box;
	-webkit-box-sizing:border-box;
	box-sizing:border-box;
	background:#fff;
	border:1px solid #A9B3BC;
	padding:8px;
	width:100%;
	min-width:550px;
	max-width:550px;
	margin-top:5px;
	outline:none;
margin-bottom:15px;
}
</style>

<script>
      var $j = jQuery.noConflict();
      var $ = jQuery.noConflict();
</script>
<script language="javascript" src="https://www.sciencemediacentre.org/wp-content/themes/SMC/tweet/jquery.tweet.js" type="text/javascript"></script>
<script>
	window.SITE_ROOT = "https://www.sciencemediacentre.org"+'/';
</script>
<script type="text/javascript" src="https://www.sciencemediacentre.org/wp-content/themes/SMC/js/SearchFilter.js"></script>
<!-- <script type="text/javascript" src="https://www.sciencemediacentre.org/wp-content/themes/SMC/js/menu.js"></script> -->

</head>
<body class="home blog wp-theme-SMC">


<!-- ALL CONTENT TO GO BELOW HERE -->
<header id="header-top">

    <div id ="browser-warning-bar">
    Our website uses the latest in web technologies. Unfortunately your browser is very old so you are missing out. We recommend you update to the latest version of IE or even better Google Chrome!</div>




<section id="header-top-content">
  <div class="mobile_nav">

    <div class="mobile_search">
      <i class="fas fa-search top_nav_container__search closed"></i>
    </div>

    <div class="hamburger_container">
      <div class="hamburger_circle">
      </div>
      <div id="hamburger_menu" class="closed">
        <span></span>
        <span></span>
        <span></span>
      </div>	
    </div>	

  </div>


<div id="header-logo">
	<div class="ImageFader">
	  <a href="https://www.sciencemediacentre.org">
      <img src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/HeaderLogo_Out.svg">
    </a>
    <a href="https://www.sciencemediacentre.org">
      <img class="top" src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/HeaderLogo_Over.svg">
    </a>
  </div>
</div><!-- header-logo -->

<section id="header-social-share-holder">
<div class="ImageFader">

  <nav id = "header-twitter">
	<a href="https://twitter.com/intent/user?screen_name=smc_london">
    <img src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/social/twitter.png" width="27" height="27">
    </a>
    <a href="https://twitter.com/intent/user?screen_name=smc_london">
    <img class="top" src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/social/twitter_over.png" width="27" height="27">
    </a>
  </nav>
  
  <nav id = "header-bebo">
	<a href="https://www.sciencemediacentre.org/blog/">
    <img src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/social/blogger.png" width="27" height="27">
    </a>
    <a href="https://www.sciencemediacentre.org/blog/">
    <img class="top" src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/social/blogger_over.png" width="27" height="27">
    </a>
  </nav>
 
  
</div>

</section><!-- header-social-share-holder -->

<section id="header-search-holder">

  <section id="search-arrow" onClick="filterArrowClicked()">

  </section><!-- search-arrow -->	


  <section id="search-input">

    <input type="text" name="searchterm" id="search-term" value=""/ onClick="OnSearchTermFocus ()">


  </section><!-- search-input-->	

  <section id="search-button" onClick="SearchBarClicked()">
  
  </section><!-- search-button-->	



</section><!-- header-search-holder -->

<section id = "header-search-filter">

<div id = "search-filter-header">select search filters</div>

<section class = "search-filter-item" onClick="BriefingsClicked()">
<div class = "search-filter-icon" id = "briefings-icon"></div>
<div class = "search-filter-label">briefings</div>
</section>

<section class = "search-filter-item" onClick="RoundUpsClicked()">
<div class = "search-filter-icon" id = "roundups-icon"></div>
<div class = "search-filter-label">roundups &amp; rapid reactions</div>
</section>

<!-- <section class = "search-filter-item" onClick="HeadlinesClicked()">
<div class = "search-filter-icon" id = "headlines-icon"></div>
<div class = "search-filter-label">before the headlines</div>
</section> -->

<section class = "search-filter-item" onClick="BlogClicked()">
<div class = "search-filter-icon" id = "blog-icon"></div>
<div class = "search-filter-label">Fiona fox's blog</div>
</section>

</section><!-- header-search-filter -->


</section><!-- header-top-content -->


<section id="top-nav-holder">
<nav id="top-nav">

<ul id="nav">
  <div class="menu-header"><ul id="menu-main_menu" class="menu"><li id="menu-item-29" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-29"><a href="https://www.sciencemediacentre.org/working-with-us/">working with us</a>
<ul class="sub-menu">
	<li id="menu-item-30" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-30"><a href="https://www.sciencemediacentre.org/working-with-us/for-journalists/">for journalists</a></li>
	<li id="menu-item-87" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-87"><a href="https://www.sciencemediacentre.org/working-with-us/for-press-officers/">for press officers</a></li>
	<li id="menu-item-88" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-88"><a href="https://www.sciencemediacentre.org/working-with-us/for-scientists/">for scientists</a></li>
</ul>
</li>
<li id="menu-item-24440" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-24440"><a href="https://www.sciencemediacentre.org/about-us/">about us</a>
<ul class="sub-menu">
	<li id="menu-item-24443" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-24443"><a href="https://www.sciencemediacentre.org/about-us/staff/">staff</a></li>
	<li id="menu-item-24442" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-24442"><a href="https://www.sciencemediacentre.org/about-us/governance/">governance</a></li>
	<li id="menu-item-24444" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-24444"><a href="https://www.sciencemediacentre.org/about-us/interns/">interns</a></li>
	<li id="menu-item-24441" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-24441"><a href="https://www.sciencemediacentre.org/about-us/funding/">funding</a></li>
	<li id="menu-item-51819" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-51819"><a href="https://www.sciencemediacentre.org/about-us/jobs/">jobs</a></li>
</ul>
</li>
<li id="menu-item-17767" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-17767"><a href="https://www.sciencemediacentre.org/publications/">publications</a>
<ul class="sub-menu">
	<li id="menu-item-25992" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-25992"><a href="https://www.sciencemediacentre.org/publications/publications-for-journalists/">publications for journalists</a></li>
	<li id="menu-item-25994" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-25994"><a href="https://www.sciencemediacentre.org/publications/publications-for-press-officers/">publications for press officers</a></li>
	<li id="menu-item-25993" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-25993"><a href="https://www.sciencemediacentre.org/publications/publications-for-scientists/">publications for scientists</a></li>
	<li id="menu-item-25999" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-25999"><a href="https://www.sciencemediacentre.org/publications/publications-about-the-science-media-centre/">publications about the Science Media Centre</a></li>
	<li id="menu-item-26000" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-26000"><a href="https://www.sciencemediacentre.org/publications/submissions-to-committees-and-inquiries/">submissions to committees and inquiries</a></li>
	<li id="menu-item-25998" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-25998"><a href="https://www.sciencemediacentre.org/publications/foundational-documents/">foundational documents</a></li>
</ul>
</li>
<li id="menu-item-17768" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-17768"><a href="https://www.sciencemediacentre.org/international-smcs/">international SMCs</a>
<ul class="sub-menu">
	<li id="menu-item-53386" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-53386"><a href="https://www.sciencemediacentre.org/ireland/">a pilot SMC for Ireland</a></li>
</ul>
</li>
<li id="menu-item-19677" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-19677"><a href="https://www.sciencemediacentre.org/contact-us/">contact us</a></li>
</ul></div>  <div class="menu-mobile"><ul id="menu-mobile-menu" class="menu"><li id="menu-item-43544" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-43544"><a>working with us</a>
<ul class="sub-menu">
	<li id="menu-item-43547" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43547"><a href="https://www.sciencemediacentre.org/working-with-us/">working with us home</a></li>
	<li id="menu-item-43548" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43548"><a href="https://www.sciencemediacentre.org/working-with-us/for-journalists/">for journalists</a></li>
	<li id="menu-item-43549" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43549"><a href="https://www.sciencemediacentre.org/working-with-us/for-press-officers/">for press officers</a></li>
	<li id="menu-item-43550" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43550"><a href="https://www.sciencemediacentre.org/working-with-us/for-scientists/">for scientists</a></li>
</ul>
</li>
<li id="menu-item-43545" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-43545"><a>about us</a>
<ul class="sub-menu">
	<li id="menu-item-43551" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43551"><a href="https://www.sciencemediacentre.org/about-us/">about us home</a></li>
	<li id="menu-item-43556" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43556"><a href="https://www.sciencemediacentre.org/about-us/staff/">staff</a></li>
	<li id="menu-item-43553" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43553"><a href="https://www.sciencemediacentre.org/about-us/governance/">governance</a></li>
	<li id="menu-item-43554" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43554"><a href="https://www.sciencemediacentre.org/about-us/interns/">interns</a></li>
	<li id="menu-item-43552" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43552"><a href="https://www.sciencemediacentre.org/about-us/funding/">funding</a></li>
	<li id="menu-item-51816" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-51816"><a href="https://www.sciencemediacentre.org/about-us/jobs/">jobs</a></li>
</ul>
</li>
<li id="menu-item-43546" class="menu-item menu-item-type-custom menu-item-object-custom menu-item-has-children menu-item-43546"><a>publications</a>
<ul class="sub-menu">
	<li id="menu-item-43557" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43557"><a href="https://www.sciencemediacentre.org/publications/">publications home</a></li>
	<li id="menu-item-43560" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43560"><a href="https://www.sciencemediacentre.org/publications/publications-for-journalists/">for journalists</a></li>
	<li id="menu-item-43561" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43561"><a href="https://www.sciencemediacentre.org/publications/publications-for-press-officers/">for press officers</a></li>
	<li id="menu-item-43562" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43562"><a href="https://www.sciencemediacentre.org/publications/publications-for-scientists/">for scientists</a></li>
	<li id="menu-item-43559" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43559"><a href="https://www.sciencemediacentre.org/publications/publications-about-the-science-media-centre/">about the Science Media Centre</a></li>
	<li id="menu-item-43563" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43563"><a href="https://www.sciencemediacentre.org/publications/submissions-to-committees-and-inquiries/">submissions to committees and inquiries</a></li>
	<li id="menu-item-43558" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43558"><a href="https://www.sciencemediacentre.org/publications/foundational-documents/">foundational documents</a></li>
</ul>
</li>
<li id="menu-item-43564" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-has-children menu-item-43564"><a href="https://www.sciencemediacentre.org/international-smcs/">international SMCs</a>
<ul class="sub-menu">
	<li id="menu-item-53473" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-53473"><a href="https://www.sciencemediacentre.org/ireland/">a pilot SMC for Ireland</a></li>
</ul>
</li>
<li id="menu-item-43565" class="menu-item menu-item-type-post_type menu-item-object-page menu-item-43565"><a href="https://www.sciencemediacentre.org/contact-us/">contact us</a></li>
</ul></div></ul>

</nav><!-- top-nav -->
</section><!-- top-nav-holder -->

</header><!-- header-top -->



<script type="text/javascript" src="https://www.sciencemediacentre.org/wp-content/themes/SMC/js/HomePage.js"></script>
<script language="javascript" src="https://www.sciencemediacentre.org/wp-content/themes/SMC/tweet/jquery.tweet.js" type="text/javascript"></script>
<link href="https://www.sciencemediacentre.org/wp-content/themes/SMC/tweet/jquery.tweet.css" media="all" rel="stylesheet" type="text/css"/>
<!-- nivo-slider -->
<!-- <link rel="stylesheet" type="text/css" media="all" href="https://www.sciencemediacentre.org/wp-content/themes/SMC/jquery.bxSlider/jquery.bxslider.css" />
<link rel="stylesheet" type="text/css" media="all" href="https://www.sciencemediacentre.org/wp-content/themes/SMC/jquery.bxSlider/css/Style.css" />
<script src="https://www.sciencemediacentre.org/wp-content/themes/SMC/jquery.bxSlider/jquery.bxslider.min.js" type="text/javascript"></script>
<script type="text/javascript">
  $(document).ready(function(){
   $('.boxSlider').bxSlider({
    auto: true,
	  controls: false,	
    pager: true,
    mode:'fade'
  });
  
   $('#intro-slider').bxSlider({
    auto: true,
	  controls: false,	
    pager: false,
    mode:'fade'

  }); -->
  
  
  
  });
</script>


<!-- QUERIES --> 
       
       
  
<!-- QUERIES --> 

<title>Science Media Center</title>

<section id="home-page-body-top-row">
  <section id="home-page-content-top-row">
    <section id = "home-page-slider-holder">
      
      <section id = "slider1-holder">
    	
        	<div id="slider1" class="boxSlider">
       
                      <!-- slide -->
            <div class="home-page-slider-container">
              <div class = "slider1-text-holder">
                <div class = "slider-title-1">
                  <h2>welcome to the</h2>
                </div>
                <div class = "slider-title-2">
                  <h1>
                    Science Media Centre                  </h1>
                </div>
                <div class = "slider-body-copy">
                  an independent press office helping to ensure that the public have access to the best scientific and engineering evidence and expertise through the news media when science hits the headlines                </div>
                <div class = "read-more-button"><a href="/about-us/">
                  find out more                  </a>
                </div>
              </div>
              <div class="slider-image-container">
                <img class="slider-image" src="https://www.sciencemediacentre.org/wp-content/uploads/2020/03/HomeBanner_3_square.jpg" /> 
              </div>
            </div>
                      <!-- slide -->
            <div class="home-page-slider-container">
              <div class = "slider1-text-holder">
                <div class = "slider-title-1">
                  <h2></h2>
                </div>
                <div class = "slider-title-2">
                  <h1>
                    Ipsos poll: where does the public access science information?                  </h1>
                </div>
                <div class = "slider-body-copy">
                                  </div>
                <div class = "read-more-button"><a href="https://www.sciencemediacentre.org/smc-press-release-news-media-a-bigger-source-of-science-information-than-social-media/">
                  read the report                  </a>
                </div>
              </div>
              <div class="slider-image-container">
                <img class="slider-image" src="https://www.sciencemediacentre.org/wp-content/uploads/2023/11/depositphotos_36242959-stock-photo-newspaper.jpg" /> 
              </div>
            </div>
                      <!-- slide -->
            <div class="home-page-slider-container">
              <div class = "slider1-text-holder">
                <div class = "slider-title-1">
                  <h2></h2>
                </div>
                <div class = "slider-title-2">
                  <h1>
                    the Science Media Centre Global Network                  </h1>
                </div>
                <div class = "slider-body-copy">
                  the new website for the Science Media Centre Global Network has launched! Visit it to learn more about the SMC Global Network - an international network ensuring the public have access to the best evidence in the news when science hits the headlines around the world.<br />
                </div>
                <div class = "read-more-button"><a href="https://www.smcglobal.org">
                  To find out more visit www.smcglobal.org                  </a>
                </div>
              </div>
              <div class="slider-image-container">
                <img class="slider-image" src="https://www.sciencemediacentre.org/wp-content/uploads/2023/03/SMC-Global-logo-carousel.png" /> 
              </div>
            </div>
                      <!-- slide -->
            <div class="home-page-slider-container">
              <div class = "slider1-text-holder">
                <div class = "slider-title-1">
                  <h2></h2>
                </div>
                <div class = "slider-title-2">
                  <h1>
                    Exciting job opportunity at the SMC!                  </h1>
                </div>
                <div class = "slider-body-copy">
                  Operations &amp; Executive Assistant                </div>
                <div class = "read-more-button"><a href="https://www.sciencemediacentre.org/about-us/jobs/job-operations-and-executive-assistant/">
                  find out more                  </a>
                </div>
              </div>
              <div class="slider-image-container">
                <img class="slider-image" src="https://www.sciencemediacentre.org/wp-content/uploads/2012/08/img3.jpg" /> 
              </div>
            </div>
                      <!-- slide -->
            <div class="home-page-slider-container">
              <div class = "slider1-text-holder">
                <div class = "slider-title-1">
                  <h2>The</h2>
                </div>
                <div class = "slider-title-2">
                  <h1>
                    Changing Role of Science Press Officers                  </h1>
                </div>
                <div class = "slider-body-copy">
                                  </div>
                <div class = "read-more-button"><a href="https://www.sciencemediacentre.org/publications/publications-for-press-officers/the-changing-role-of-science-press-officers/">
                  read the report                  </a>
                </div>
              </div>
              <div class="slider-image-container">
                <img class="slider-image" src="https://www.sciencemediacentre.org/wp-content/uploads/2019/01/Website-Image-2.png" /> 
              </div>
            </div>
                  </div>
        <!-- slider1 --> 
      </section>
      <!-- slider1-holder -->
      
      
       	      <nav id = "pager"></nav>
 		        
     
    </section>
    <!-- home-page-slider-holder -->
   
    <div class="homepage-h-divider"></div>
    <section class="home-page-header-column-container">
      <section class="home-page-header-column">
        <h1 class="home-page-header-h1">For Journalists</h1>
      
    
        <div class = "home-page-header-column-body-text">
          The Science Media Centre provides journalists with what they need in the timeframe they need it, from interviews with leading experts to timely press briefings on hot topics.        </div>
        <nav class = "home-page-header-column-button"><a href="https://www.sciencemediacentre.org/working-with-us/for-journalists/">find out more</a></nav>
      </section>
      
      
      <section class="home-page-header-column">
        <h1 class="home-page-header-h1">for scientists</h1>
        <div class = "home-page-header-column-body-text">
        The SMC runs free events to introduce experts to the news media.  We also provide advice and support to scientists and engineers on media engagement.        </div>
        <nav class = "home-page-header-column-button"><a href="https://www.sciencemediacentre.org/working-with-us/for-scientists/">find out more</a></nav>
      </section>
      
      
      <section class="home-page-header-column" id = "last-home-column">
        <h1 class="home-page-header-h1">for press officers</h1>
        <div class = "home-page-header-column-body-text">
        The SMC is an independent press office for science, working closely with press officers from universities, scientific companies, research funders and leading science and engineering institutions.        </div>
        <nav class = "home-page-header-column-button"><a href="https://www.sciencemediacentre.org/working-with-us/for-press-officers/">find out more</a></nav>
      </section>
    
    </section>
    
  </section>
  <!-- home-page-content-top-row --> 
  
</section>
<!-- home-page-body-top-row -->

<section id="home-page-body-mid-row">
  <section id="home-page-content-mid-row">
    <section id="home-page-left-column">
      <nav id="filter-by-nav">

    <section class="filter-by-item">
        <div class = "filter-by-image-holder" id = "filter-roundups">
            <div class = "icon-out">
                <a href="https://www.sciencemediacentre.org/working-with-us/for-journalists/roundups-for-journalists/">
                    <img src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/Home_RoundUps_Out.png"  />
                </a>
            </div>
            <div class = "icon-over">
                <a href="https://www.sciencemediacentre.org/working-with-us/for-journalists/roundups-for-journalists/">
                    <img src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/Home_RoundUps_Over.png"  />
                </a>
            </div>
        </div>
        <h1 class = "filter-by-title">
            <a href="https://www.sciencemediacentre.org/working-with-us/for-journalists/roundups-for-journalists/">roundups & rapid reactions</a>
        </h1>
    </section>

    <section class="filter-by-item">
        <div class = "filter-by-image-holder" id = "filter-briefings">
            <div class = "icon-out">
                <a href="https://www.sciencemediacentre.org/working-with-us/for-journalists/briefings-for-journalists/">
                    <img src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/Home_Briefings_Out.png"  />
                </a>
            </div>
            <div class = "icon-over">
                <a href="https://www.sciencemediacentre.org/working-with-us/for-journalists/briefings-for-journalists/">
                    <img src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/Home_Briefings_Over.png"  />
                </a>
            </div>
        </div>
        <h1 class = "filter-by-title">
            <a href="https://www.sciencemediacentre.org/working-with-us/for-journalists/briefings-for-journalists/">briefings</a>
        </h1>
    </section>

    <!-- <section class="filter-by-item">
        <div class = "filter-by-image-holder" id = "filter-factsheets">
            <div class = "icon-out">
                <a href="https://www.sciencemediacentre.org/working-with-us/for-journalists/fact-sheets-for-journalists/">
                    <img src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/Home_FactSheets_Out.png"  />
                </a>
            </div>
            <div class = "icon-over">
                <a href="https://www.sciencemediacentre.org/working-with-us/for-journalists/fact-sheets-for-journalists/">
                    <img src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/Home_FactSheets_Over.png"  />
                </a>
            </div>
        </div>
        <h1 class = "filter-by-title">
            <a href="https://www.sciencemediacentre.org/working-with-us/for-journalists/fact-sheets-for-journalists/">factsheets &amp briefing notes</a>
        </h1>
    </section> -->

    <!-- <section class="filter-by-item">
        <div class = "filter-by-image-holder" id = "filter-headlines">
            <div class = "icon-out">
                <a href="https://www.sciencemediacentre.org/working-with-us/for-journalists/headlines-for-journalists/">
                    <img src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/Home_Headlines_Out.png"  />
                </a>
            </div>
            <div class = "icon-over">
                <a href="https://www.sciencemediacentre.org/working-with-us/for-journalists/headlines-for-journalists/">
                    <img src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/Home_Headlines_Over.png"  />
                </a>
            </div>
        </div>
        <h1 class = "filter-by-title">
            <a href="https://www.sciencemediacentre.org/working-with-us/for-journalists/headlines-for-journalists/">before the headlines</a>
        </h1>
    </section> -->

</nav>      <div class="left-column-h-divider"></div>
            <section id="article-list">
                            <article class = "article-list-item">
            <a href="https://www.sciencemediacentre.org/expert-reaction-to-first-complete-sequence-of-the-hantavirus-from-the-current-cluster-from-mv-hondius-from-the-swiss-patient-with-confirmed-andes-strain-uploaded-to-the-virological-org-platform-by-t/"><span class = "article-list-item-span"></span></a>
            <section class = "article-list-image-continer">
            <div class = "article-list-image image-over article-list-image-over-roundups"></div> 
            <div class = "article-list-image image-out article-list-image-out-roundups"></div> 
            </section>
            <section class = "article-list-text-continer">
              <header>
                <h3><time pubdate="pubdate">May 9, 2026</time></h3>
                <h2>
                <a href="https://www.sciencemediacentre.org/expert-reaction-to-first-complete-sequence-of-the-hantavirus-from-the-current-cluster-from-mv-hondius-from-the-swiss-patient-with-confirmed-andes-strain-uploaded-to-the-virological-org-platform-by-t/">expert reaction to first complete sequence of the hantavirus from the current cluster from MV Hondius (from the Swiss patient with confirmed Andes strain) uploaded to the Virological.org platform by the Swiss National Reference Center for Emerging Viral Infections, Geneva University Hospitals and the Institute of Medical Virology, University of Zurich</a>
                </h2>
              </header>
              <p>Scientists gave comments on the first complete sequence of the hantavirus from the current cluster related to the MV Hondius &hellip; <a href="https://www.sciencemediacentre.org/expert-reaction-to-first-complete-sequence-of-the-hantavirus-from-the-current-cluster-from-mv-hondius-from-the-swiss-patient-with-confirmed-andes-strain-uploaded-to-the-virological-org-platform-by-t/">read more</a></p>
            </section>
          </article>
      
      
                            <article class = "article-list-item">
            <a href="https://www.sciencemediacentre.org/expert-reaction-to-statement-from-the-ukhsa-dhsc-and-fcdo-re-hantavirus-including-transport-arrangements-and-isolation-for-british-passengers-and-crew-aboard-the-mv-hondius/"><span class = "article-list-item-span"></span></a>
            <section class = "article-list-image-continer">
            <div class = "article-list-image image-over article-list-image-over-roundups"></div> 
            <div class = "article-list-image image-out article-list-image-out-roundups"></div> 
            </section>
            <section class = "article-list-text-continer">
              <header>
                <h3><time pubdate="pubdate">May 8, 2026</time></h3>
                <h2>
                <a href="https://www.sciencemediacentre.org/expert-reaction-to-statement-from-the-ukhsa-dhsc-and-fcdo-re-hantavirus-including-transport-arrangements-and-isolation-for-british-passengers-and-crew-aboard-the-mv-hondius/">expert reaction to statement from the UKHSA, DHSC and FCDO re hantavirus, including transport arrangements and isolation for British passengers and crew aboard the MV Hondius</a>
                </h2>
              </header>
              <p>Scientists comment on a statement from the UK Health Security Agency (UKHSA) on hantavirus, travel arrangements and isolation for British &hellip; <a href="https://www.sciencemediacentre.org/expert-reaction-to-statement-from-the-ukhsa-dhsc-and-fcdo-re-hantavirus-including-transport-arrangements-and-isolation-for-british-passengers-and-crew-aboard-the-mv-hondius/">read more</a></p>
            </section>
          </article>
      
      
                            <article class = "article-list-item">
            <a href="https://www.sciencemediacentre.org/expert-reaction-to-hantavirus-update-from-the-ukhsa-including-that-two-people-had-returned-to-the-uk-having-been-on-board-the-mv-hondius-are-not-currently-reporting-symptoms-and-are-self-isolating/"><span class = "article-list-item-span"></span></a>
            <section class = "article-list-image-continer">
            <div class = "article-list-image image-over article-list-image-over-roundups"></div> 
            <div class = "article-list-image image-out article-list-image-out-roundups"></div> 
            </section>
            <section class = "article-list-text-continer">
              <header>
                <h3><time pubdate="pubdate">May 7, 2026</time></h3>
                <h2>
                <a href="https://www.sciencemediacentre.org/expert-reaction-to-hantavirus-update-from-the-ukhsa-including-that-two-people-had-returned-to-the-uk-having-been-on-board-the-mv-hondius-are-not-currently-reporting-symptoms-and-are-self-isolating/">expert reaction to hantavirus update from the UKHSA, including that two people had returned to the UK having been on board the MV Hondius, are not currently reporting symptoms and are self-isolating</a>
                </h2>
              </header>
              <p>Scientists comment on a hantavirus update from the UK Health Security Agency (UKHSA) that two people from the UK are &hellip; <a href="https://www.sciencemediacentre.org/expert-reaction-to-hantavirus-update-from-the-ukhsa-including-that-two-people-had-returned-to-the-uk-having-been-on-board-the-mv-hondius-are-not-currently-reporting-symptoms-and-are-self-isolating/">read more</a></p>
            </section>
          </article>
      
      
                            <article class = "article-list-item">
            <a href="https://www.sciencemediacentre.org/expert-reaction-to-clinical-consensus-statement-on-ultra-processed-foods-and-cardiovascular-diseases/"><span class = "article-list-item-span"></span></a>
            <section class = "article-list-image-continer">
            <div class = "article-list-image image-over article-list-image-over-roundups"></div> 
            <div class = "article-list-image image-out article-list-image-out-roundups"></div> 
            </section>
            <section class = "article-list-text-continer">
              <header>
                <h3><time pubdate="pubdate">May 7, 2026</time></h3>
                <h2>
                <a href="https://www.sciencemediacentre.org/expert-reaction-to-clinical-consensus-statement-on-ultra-processed-foods-and-cardiovascular-diseases/">expert reaction to clinical consensus statement on ultra-processed foods and cardiovascular diseases</a>
                </h2>
              </header>
              <p>A clinical consensus statement published in the European Heart Journal looks at UPFs and cardiovascular diseases.    Dr Oonagh Markey, Senior Lecturer &hellip; <a href="https://www.sciencemediacentre.org/expert-reaction-to-clinical-consensus-statement-on-ultra-processed-foods-and-cardiovascular-diseases/">read more</a></p>
            </section>
          </article>
      
      
                            <article class = "article-list-item">
            <a href="https://www.sciencemediacentre.org/expert-reaction-to-systematic-review-and-meta-analysis-on-ketamine-infusions-for-rapid-reduction-of-suicidal-and-depressive-symptoms-in-treating-major-depressive-episodes-mdes/"><span class = "article-list-item-span"></span></a>
            <section class = "article-list-image-continer">
            <div class = "article-list-image image-over article-list-image-over-roundups"></div> 
            <div class = "article-list-image image-out article-list-image-out-roundups"></div> 
            </section>
            <section class = "article-list-text-continer">
              <header>
                <h3><time pubdate="pubdate">May 6, 2026</time></h3>
                <h2>
                <a href="https://www.sciencemediacentre.org/expert-reaction-to-systematic-review-and-meta-analysis-on-ketamine-infusions-for-rapid-reduction-of-suicidal-and-depressive-symptoms-in-treating-major-depressive-episodes-mdes/">expert reaction to systematic review and meta-analysis on ketamine infusions for rapid reduction of suicidal and depressive symptoms in treating major depressive episodes (MDEs)</a>
                </h2>
              </header>
              <p>A systematic review and meta-analysis looks at ketamine infusions for the reductio of suicidal and depressive symptoms.    Prof Rupert &hellip; <a href="https://www.sciencemediacentre.org/expert-reaction-to-systematic-review-and-meta-analysis-on-ketamine-infusions-for-rapid-reduction-of-suicidal-and-depressive-symptoms-in-treating-major-depressive-episodes-mdes/">read more</a></p>
            </section>
          </article>
      
      
                            <article class = "article-list-item">
            <a href="https://www.sciencemediacentre.org/expert-reaction-to-continuing-hantavirus-situation-aboard-cruise-ship/"><span class = "article-list-item-span"></span></a>
            <section class = "article-list-image-continer">
            <div class = "article-list-image image-over article-list-image-over-roundups"></div> 
            <div class = "article-list-image image-out article-list-image-out-roundups"></div> 
            </section>
            <section class = "article-list-text-continer">
              <header>
                <h3><time pubdate="pubdate">May 6, 2026</time></h3>
                <h2>
                <a href="https://www.sciencemediacentre.org/expert-reaction-to-continuing-hantavirus-situation-aboard-cruise-ship/">expert reaction to continuing hantavirus situation aboard cruise ship</a>
                </h2>
              </header>
              <p>Scientists comment on the continuing hantavirus situation aboard a cruise ship.    Dr Giulia Gallo, Postdoctoral Scientist in the Viral &hellip; <a href="https://www.sciencemediacentre.org/expert-reaction-to-continuing-hantavirus-situation-aboard-cruise-ship/">read more</a></p>
            </section>
          </article>
      
      
                            <article class = "article-list-item">
            <a href="https://www.sciencemediacentre.org/expert-reaction-to-study-of-at-home-blood-test-and-cognitive-test-for-alzheimers-risk/"><span class = "article-list-item-span"></span></a>
            <section class = "article-list-image-continer">
            <div class = "article-list-image image-over article-list-image-over-roundups"></div> 
            <div class = "article-list-image image-out article-list-image-out-roundups"></div> 
            </section>
            <section class = "article-list-text-continer">
              <header>
                <h3><time pubdate="pubdate">May 6, 2026</time></h3>
                <h2>
                <a href="https://www.sciencemediacentre.org/expert-reaction-to-study-of-at-home-blood-test-and-cognitive-test-for-alzheimers-risk/">expert reaction to study of at home blood test and cognitive test for Alzheimer’s risk</a>
                </h2>
              </header>
              <p>A study published in Nature Communications looks at an at home blood test for Alzheimer&#8217;s disease biomarkers.    Dr Amanda Heslegrave, Principal &hellip; <a href="https://www.sciencemediacentre.org/expert-reaction-to-study-of-at-home-blood-test-and-cognitive-test-for-alzheimers-risk/">read more</a></p>
            </section>
          </article>
      
      
              </section>
      <!-- artical-list --> 
  
    </section>
    <!-- home-page-left-column -->
    
    <section class="home-page-right-side-bar">
    
      <section id = "blog-entry">
                <a href="https://www.sciencemediacentre.org/blog/">
          <h1>read Fiona Fox’s blog</h1>
          <h2>on science and the media</h2>
        </a>
                          
            <a href="https://www.sciencemediacentre.org/what-the-public-attitudes-to-science-survey-is-telling-us-about-communication-in-an-era-of-misinformation/"><span class = "blog-entry-span"></span></a>
            <div class="blog-entry-container">    
              <div id = "Polaroid">
                <img src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/FionaFox-Avatar-image.jpg" /></div>
              <div id = "blog-entry-text-holder">
                <div id= "blog-entry-text">
                  latest blog entry: "Science communication in an era of misinformation – what the Public Attitudes to Science survey is telling us"
                                  </div>
                <!--<h3>
                  <time pubdate="pubdate">March 19, 2026</time>
                </h3>-->
                <div class = "read-more-button-sidebar"><a href="https://www.sciencemediacentre.org/what-the-public-attitudes-to-science-survey-is-telling-us-about-communication-in-an-era-of-misinformation/">read more</a></div>     
              </div>
            </div>
                  
        <div class="homepage-right-column-h-divider"></div>
      </section> <!-- blog-entry --> 
    
      <section id = "tweet-holder" class="homepage_tweet_holder">
        <div class="tweet"></div>
      </section>
    </section>
    <!-- home-page-right-side-bar --> 
    
  </section>
  <!-- home-page-content-mid-row --> 
  
</section>
<!-- home-page-body-mid-row -->

<section id="home-page-body-bottom-row">
  <section id="home-page-content-bottom-row">
    
    <section id="for-the-public">
      <h1>for the public</h1>
      <p>The Science Media Centre is a service for national news journalists. If you are a member of the public, and have a question about science, you could try:</p>
                <div class="like-link-button-container">
                      <nav class="like-link-button">
              <a href="http://www.nhs.uk" target="_blank">
                <img src="https://www.sciencemediacentre.org/wp-content/uploads/2012/08/nhs_choices-e1612436542775.png" width="290" height="75" />
              </a> 
            </nav>
                      <nav class="like-link-button">
              <a href="https://fullfact.org/" target="_blank">
                <img src="https://www.sciencemediacentre.org/wp-content/uploads/2012/08/Full-fact-logo.png" width="290" height="75" />
              </a> 
            </nav>
                      <nav class="like-link-button">
              <a href="http://www.britishscienceassociation.org" target="_blank">
                <img src="https://www.sciencemediacentre.org/wp-content/uploads/2020/02/BSA-Logo_Mono-768x270-1.png" width="290" height="75" />
              </a> 
            </nav>
                      <nav class="like-link-button">
              <a href="http://www.senseaboutscience.org" target="_blank">
                <img src="https://www.sciencemediacentre.org/wp-content/uploads/2024/09/SaS_linear_logo_with_strap_RGB_50x60mm-1-1.png" width="290" height="75" />
              </a> 
            </nav>
                  </div>
    </section>  <!-- for-the-public --> 
    
  </section>
  <!-- home-page-content-bottom-row --> 
  
</section>
<!-- home-page-body-bottom-row -->


<footer id="footer-bot">
  <section id="footer-bot-content">
    <section id="footer-col-left"> <img class="footer_logo" src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/FooterLogo.png" width="279" height="62" />
      <section id="footer-address"> 215 Euston Road, London, NW1 2BE<br>
        <img class="email-footer-image" src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/email-footer.png"><br>
        +44 (0)20 7611 8300 </section>
      <!-- footer-address -->
      
      <div id="terms-policy">
      
      <a href="https://www.sciencemediacentre.org/terms-and-conditions">Terms and Conditions</a> | <a href="https://www.sciencemediacentre.org/privacy">Privacy Notice</a> </div>
      <!-- Terms & Conditions + Privacy Notice --> 
      
      
    </section>
    <!-- footer-col-left -->
    
    <section id="footer-col-middle">

      <section class="footer-social-share-holder">
        <a href="https://twitter.com/intent/user?screen_name=smc_london">
    	<nav id="footer-twitter"><img src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/social_transparent.png" width="27" height="27"></nav>
      	<span class="footer-social-text">follow us on twitter</span></a>  	
      </section>
    
      <section class="footer-social-share-holder"> 
          <a href="https://www.sciencemediacentre.org/blog/">
          <nav id="footer-blogger">
          <img src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/social_transparent.png" width="27" height="27">
          </nav>
          <span class="footer-social-text">read Fiona's blog</span></a>
      </section>
      
      <aside id="footer-image-creds">With thanks to <a href="https://wellcomecollection.org/collections" target="_blank">Wellcome Collection</a> for use of their image library</aside>
      <!-- footer-image-creds --> 
      
  
    </section>
    <!-- footer-col-middle -->
    
    <section id="footer-col-right">
      <aside id="footer-small-print">Science Media Centre is a registered charity No. 1140827<br>
      and a company limited by guarantee, No. 7560997<br><br>
      Registered in England and Wales.<br><br>
      All worldwide rights reserved for all content on this site.<br>
      Copyright © Science Media Centre 2012</aside>
      <!-- footer-small-print --> 
      
      <section id="footer-by-node">
          <div id="footer-by-node-image">
            <div class="ImageFader">
              <a href="http://www.nodelondon.com" target="_blank">
              <img src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/FooterNode.png" width="193" height="78" />
              </a>
              <a href="http://www.nodelondon.com" target="_blank">
              <img class="top" src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/FooterNode_Over.png" width="193" height="78" />
              </a>
            </div>
          </div>
          
        <div id="footer-by-node-text">
        	constructed on a molecular level by node
        </div>
        
        <div id="servedBy">
        	hosted by<a href="http://www.mythic-beasts.com" target="_blank">
            	<img class="top" src="https://www.sciencemediacentre.org/wp-content/themes/SMC/images/mythic-beasts-logo.png" width="139" height="35" />
            </a>
        </div>
      
      </section><!-- footer-by-node --> 
      
    </section>
    <!-- footer-col-right --> 
  </section>
  <!-- footer-bot-content --> 
  
</footer>
<!-- footer-bot  -->

<script type="text/javascript" src="//platform.twitter.com/widgets.js"></script>
<script type="text/javascript">

  var _gaq = _gaq || [];
  _gaq.push(['_setAccount', 'UA-34163666-1']);
  _gaq.push(['_trackPageview']);

  (function() {
    var ga = document.createElement('script'); ga.type = 'text/javascript'; ga.async = true;
    ga.src = ('https:' == document.location.protocol ? 'https://ssl' : 'http://www') + '.google-analytics.com/ga.js';
    var s = document.getElementsByTagName('script')[0]; s.parentNode.insertBefore(ga, s);
  })();

</script>

<script language="javascript" src="https://www.sciencemediacentre.org/wp-content/themes/SMC/assets/build/app.js" type="text/javascript"></script>


<!--Plugin Global Hide Admin Tool Bar Active-->

<!-- This website is patched against a big problem not solved from WordPress 3.3+ to date -->

<script type="speculationrules">
{"prefetch":[{"source":"document","where":{"and":[{"href_matches":"/*"},{"not":{"href_matches":["/wp-*.php","/wp-admin/*","/wp-content/uploads/*","/wp-content/*","/wp-content/plugins/*","/wp-content/themes/SMC/*","/*\\?(.+)"]}},{"not":{"selector_matches":"a[rel~=\"nofollow\"]"}},{"not":{"selector_matches":".no-prefetch, .no-prefetch a"}}]},"eagerness":"conservative"}]}
</script>
<!--googleoff: all--><div id="cookie-law-info-bar" data-nosnippet="true"><span>This website uses only necessary cookies, to improve your experience. <a href="http://www.sciencemediacentre.org/terms-and-conditions/" id="CONSTANT_OPEN_URL" target="_blank" class="cli-plugin-main-link">Read More</a> <a role='button' data-cli_action="accept" id="cookie_action_close_header" class="medium cli-plugin-button cli-plugin-main-button cookie_action_close_header cli_action_button wt-cli-accept-btn">Accept</a></span></div><div id="cookie-law-info-again" data-nosnippet="true"><span id="cookie_hdr_showagain"></span></div><div class="cli-modal" data-nosnippet="true" id="cliSettingsPopup" tabindex="-1" role="dialog" aria-labelledby="cliSettingsPopup" aria-hidden="true">
  <div class="cli-modal-dialog" role="document">
	<div class="cli-modal-content cli-bar-popup">
		  <button type="button" class="cli-modal-close" id="cliModalClose">
			<svg class="" viewBox="0 0 24 24"><path d="M19 6.41l-1.41-1.41-5.59 5.59-5.59-5.59-1.41 1.41 5.59 5.59-5.59 5.59 1.41 1.41 5.59-5.59 5.59 5.59 1.41-1.41-5.59-5.59z"></path><path d="M0 0h24v24h-24z" fill="none"></path></svg>
			<span class="wt-cli-sr-only">Close</span>
		  </button>
		  <div class="cli-modal-body">
			<div class="cli-container-fluid cli-tab-container">
	<div class="cli-row">
		<div class="cli-col-12 cli-align-items-stretch cli-px-0">
			<div class="cli-privacy-overview">
				<h4>Privacy Overview</h4>				<div class="cli-privacy-content">
					<div class="cli-privacy-content-text">This website uses cookies to improve your experience while you navigate through the website. Out of these, the cookies that are categorized as necessary are stored on your browser as they are essential for the working of basic functionalities of the website. We also use third-party cookies that help us analyze and understand how you use this website. These cookies will be stored in your browser only with your consent. You also have the option to opt-out of these cookies. But opting out of some of these cookies may affect your browsing experience.</div>
				</div>
				<a class="cli-privacy-readmore" aria-label="Show more" role="button" data-readmore-text="Show more" data-readless-text="Show less"></a>			</div>
		</div>
		<div class="cli-col-12 cli-align-items-stretch cli-px-0 cli-tab-section-container">
												<div class="cli-tab-section">
						<div class="cli-tab-header">
							<a role="button" tabindex="0" class="cli-nav-link cli-settings-mobile" data-target="necessary" data-toggle="cli-toggle-tab">
								Necessary							</a>
															<div class="wt-cli-necessary-checkbox">
									<input type="checkbox" class="cli-user-preference-checkbox"  id="wt-cli-checkbox-necessary" data-id="checkbox-necessary" checked="checked"  />
									<label class="form-check-label" for="wt-cli-checkbox-necessary">Necessary</label>
								</div>
								<span class="cli-necessary-caption">Always Enabled</span>
													</div>
						<div class="cli-tab-content">
							<div class="cli-tab-pane cli-fade" data-id="necessary">
								<div class="wt-cli-cookie-description">
									Necessary cookies are absolutely essential for the website to function properly. This category only includes cookies that ensures basic functionalities and security features of the website. These cookies do not store any personal information.								</div>
							</div>
						</div>
					</div>
																	<div class="cli-tab-section">
						<div class="cli-tab-header">
							<a role="button" tabindex="0" class="cli-nav-link cli-settings-mobile" data-target="non-necessary" data-toggle="cli-toggle-tab">
								Non-necessary							</a>
															<div class="cli-switch">
									<input type="checkbox" id="wt-cli-checkbox-non-necessary" class="cli-user-preference-checkbox"  data-id="checkbox-non-necessary" checked='checked' />
									<label for="wt-cli-checkbox-non-necessary" class="cli-slider" data-cli-enable="Enabled" data-cli-disable="Disabled"><span class="wt-cli-sr-only">Non-necessary</span></label>
								</div>
													</div>
						<div class="cli-tab-content">
							<div class="cli-tab-pane cli-fade" data-id="non-necessary">
								<div class="wt-cli-cookie-description">
									Any cookies that may not be particularly necessary for the website to function and is used specifically to collect user personal data via analytics, ads, other embedded contents are termed as non-necessary cookies. It is mandatory to procure user consent prior to running these cookies on your website.								</div>
							</div>
						</div>
					</div>
										</div>
	</div>
</div>
		  </div>
		  <div class="cli-modal-footer">
			<div class="wt-cli-element cli-container-fluid cli-tab-container">
				<div class="cli-row">
					<div class="cli-col-12 cli-align-items-stretch cli-px-0">
						<div class="cli-tab-footer wt-cli-privacy-overview-actions">
						
															<a id="wt-cli-privacy-save-btn" role="button" tabindex="0" data-cli-action="accept" class="wt-cli-privacy-btn cli_setting_save_button wt-cli-privacy-accept-btn cli-btn">SAVE &amp; ACCEPT</a>
													</div>
						
					</div>
				</div>
			</div>
		</div>
	</div>
  </div>
</div>
<div class="cli-modal-backdrop cli-fade cli-settings-overlay"></div>
<div class="cli-modal-backdrop cli-fade cli-popupbar-overlay"></div>
<!--googleon: all--><script id="wp-emoji-settings" type="application/json">
{"baseUrl":"https://s.w.org/images/core/emoji/17.0.2/72x72/","ext":".png","svgUrl":"https://s.w.org/images/core/emoji/17.0.2/svg/","svgExt":".svg","source":{"concatemoji":"https://www.sciencemediacentre.org/wp-includes/js/wp-emoji-release.min.js?ver=6.9.4"}}
</script>
<script type="module">
/* <![CDATA[ */
/*! This file is auto-generated */
const a=JSON.parse(document.getElementById("wp-emoji-settings").textContent),o=(window._wpemojiSettings=a,"wpEmojiSettingsSupports"),s=["flag","emoji"];function i(e){try{var t={supportTests:e,timestamp:(new Date).valueOf()};sessionStorage.setItem(o,JSON.stringify(t))}catch(e){}}function c(e,t,n){e.clearRect(0,0,e.canvas.width,e.canvas.height),e.fillText(t,0,0);t=new Uint32Array(e.getImageData(0,0,e.canvas.width,e.canvas.height).data);e.clearRect(0,0,e.canvas.width,e.canvas.height),e.fillText(n,0,0);const a=new Uint32Array(e.getImageData(0,0,e.canvas.width,e.canvas.height).data);return t.every((e,t)=>e===a[t])}function p(e,t){e.clearRect(0,0,e.canvas.width,e.canvas.height),e.fillText(t,0,0);var n=e.getImageData(16,16,1,1);for(let e=0;e<n.data.length;e++)if(0!==n.data[e])return!1;return!0}function u(e,t,n,a){switch(t){case"flag":return n(e,"\ud83c\udff3\ufe0f\u200d\u26a7\ufe0f","\ud83c\udff3\ufe0f\u200b\u26a7\ufe0f")?!1:!n(e,"\ud83c\udde8\ud83c\uddf6","\ud83c\udde8\u200b\ud83c\uddf6")&&!n(e,"\ud83c\udff4\udb40\udc67\udb40\udc62\udb40\udc65\udb40\udc6e\udb40\udc67\udb40\udc7f","\ud83c\udff4\u200b\udb40\udc67\u200b\udb40\udc62\u200b\udb40\udc65\u200b\udb40\udc6e\u200b\udb40\udc67\u200b\udb40\udc7f");case"emoji":return!a(e,"\ud83e\u1fac8")}return!1}function f(e,t,n,a){let r;const o=(r="undefined"!=typeof WorkerGlobalScope&&self instanceof WorkerGlobalScope?new OffscreenCanvas(300,150):document.createElement("canvas")).getContext("2d",{willReadFrequently:!0}),s=(o.textBaseline="top",o.font="600 32px Arial",{});return e.forEach(e=>{s[e]=t(o,e,n,a)}),s}function r(e){var t=document.createElement("script");t.src=e,t.defer=!0,document.head.appendChild(t)}a.supports={everything:!0,everythingExceptFlag:!0},new Promise(t=>{let n=function(){try{var e=JSON.parse(sessionStorage.getItem(o));if("object"==typeof e&&"number"==typeof e.timestamp&&(new Date).valueOf()<e.timestamp+604800&&"object"==typeof e.supportTests)return e.supportTests}catch(e){}return null}();if(!n){if("undefined"!=typeof Worker&&"undefined"!=typeof OffscreenCanvas&&"undefined"!=typeof URL&&URL.createObjectURL&&"undefined"!=typeof Blob)try{var e="postMessage("+f.toString()+"("+[JSON.stringify(s),u.toString(),c.toString(),p.toString()].join(",")+"));",a=new Blob([e],{type:"text/javascript"});const r=new Worker(URL.createObjectURL(a),{name:"wpTestEmojiSupports"});return void(r.onmessage=e=>{i(n=e.data),r.terminate(),t(n)})}catch(e){}i(n=f(s,u,c,p))}t(n)}).then(e=>{for(const n in e)a.supports[n]=e[n],a.supports.everything=a.supports.everything&&a.supports[n],"flag"!==n&&(a.supports.everythingExceptFlag=a.supports.everythingExceptFlag&&a.supports[n]);var t;a.supports.everythingExceptFlag=a.supports.everythingExceptFlag&&!a.supports.flag,a.supports.everything||((t=a.source||{}).concatemoji?r(t.concatemoji):t.wpemoji&&t.twemoji&&(r(t.twemoji),r(t.wpemoji)))});
//# sourceURL=https://www.sciencemediacentre.org/wp-includes/js/wp-emoji-loader.min.js
/* ]]> */
</script>





</body></html>
<!-- Dynamic page generated in 0.329 seconds. -->
<!-- Cached page generated by WP-Super-Cache on 2026-05-10 11:05:35 -->

<!-- super cache -->
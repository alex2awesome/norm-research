<!-- URL: https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/ -->
<!-- HTTP: 200 SIZE:    99052 -->
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
	<title>Barcelona Principles 3.0 &#8211; communications evaluation | intelligent measurement</title>
	<link rel="pingback" href="https://intelligentmeasurement.net/xmlrpc.php" />
	<meta name='robots' content='max-image-preview:large' />

<!-- Async WordPress.com Remote Login -->
<script id="wpcom_remote_login_js">
var wpcom_remote_login_extra_auth = '';
function wpcom_remote_login_remove_dom_node_id( element_id ) {
	var dom_node = document.getElementById( element_id );
	if ( dom_node ) { dom_node.parentNode.removeChild( dom_node ); }
}
function wpcom_remote_login_remove_dom_node_classes( class_name ) {
	var dom_nodes = document.querySelectorAll( '.' + class_name );
	for ( var i = 0; i < dom_nodes.length; i++ ) {
		dom_nodes[ i ].parentNode.removeChild( dom_nodes[ i ] );
	}
}
function wpcom_remote_login_final_cleanup() {
	wpcom_remote_login_remove_dom_node_classes( "wpcom_remote_login_msg" );
	wpcom_remote_login_remove_dom_node_id( "wpcom_remote_login_key" );
	wpcom_remote_login_remove_dom_node_id( "wpcom_remote_login_validate" );
	wpcom_remote_login_remove_dom_node_id( "wpcom_remote_login_js" );
	wpcom_remote_login_remove_dom_node_id( "wpcom_request_access_iframe" );
	wpcom_remote_login_remove_dom_node_id( "wpcom_request_access_styles" );
}

// Watch for messages back from the remote login
window.addEventListener( "message", function( e ) {
	if ( e.origin === "https://r-login.wordpress.com" ) {
		var data = {};
		try {
			data = JSON.parse( e.data );
		} catch( e ) {
			wpcom_remote_login_final_cleanup();
			return;
		}

		if ( data.msg === 'LOGIN' ) {
			// Clean up the login check iframe
			wpcom_remote_login_remove_dom_node_id( "wpcom_remote_login_key" );

			var id_regex = new RegExp( /^[0-9]+$/ );
			var token_regex = new RegExp( /^.*|.*|.*$/ );
			if (
				token_regex.test( data.token )
				&& id_regex.test( data.wpcomid )
			) {
				// We have everything we need to ask for a login
				var script = document.createElement( "script" );
				script.setAttribute( "id", "wpcom_remote_login_validate" );
				script.src = '/remote-login.php?wpcom_remote_login=validate'
					+ '&wpcomid=' + data.wpcomid
					+ '&token=' + encodeURIComponent( data.token )
					+ '&host=' + window.location.protocol
					+ '//' + window.location.hostname
					+ '&postid=2103'
					+ '&is_singular=1';
				document.body.appendChild( script );
			}

			return;
		}

		// Safari ITP, not logged in, so redirect
		if ( data.msg === 'LOGIN-REDIRECT' ) {
			window.location = 'https://wordpress.com/log-in?redirect_to=' + window.location.href;
			return;
		}

		// Safari ITP, storage access failed, remove the request
		if ( data.msg === 'LOGIN-REMOVE' ) {
			var css_zap = 'html { -webkit-transition: margin-top 1s; transition: margin-top 1s; } /* 9001 */ html { margin-top: 0 !important; } * html body { margin-top: 0 !important; } @media screen and ( max-width: 782px ) { html { margin-top: 0 !important; } * html body { margin-top: 0 !important; } }';
			var style_zap = document.createElement( 'style' );
			style_zap.type = 'text/css';
			style_zap.appendChild( document.createTextNode( css_zap ) );
			document.body.appendChild( style_zap );

			var e = document.getElementById( 'wpcom_request_access_iframe' );
			e.parentNode.removeChild( e );

			document.cookie = 'wordpress_com_login_access=denied; path=/; max-age=31536000';

			return;
		}

		// Safari ITP
		if ( data.msg === 'REQUEST_ACCESS' ) {
			console.log( 'request access: safari' );

			// Check ITP iframe enable/disable knob
			if ( wpcom_remote_login_extra_auth !== 'safari_itp_iframe' ) {
				return;
			}

			// If we are in a "private window" there is no ITP.
			var private_window = false;
			try {
				var opendb = window.openDatabase( null, null, null, null );
			} catch( e ) {
				private_window = true;
			}

			if ( private_window ) {
				console.log( 'private window' );
				return;
			}

			var iframe = document.createElement( 'iframe' );
			iframe.id = 'wpcom_request_access_iframe';
			iframe.setAttribute( 'scrolling', 'no' );
			iframe.setAttribute( 'sandbox', 'allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-top-navigation-by-user-activation' );
			iframe.src = 'https://r-login.wordpress.com/remote-login.php?wpcom_remote_login=request_access&origin=' + encodeURIComponent( data.origin ) + '&wpcomid=' + encodeURIComponent( data.wpcomid );

			var css = 'html { -webkit-transition: margin-top 1s; transition: margin-top 1s; } /* 9001 */ html { margin-top: 46px !important; } * html body { margin-top: 46px !important; } @media screen and ( max-width: 660px ) { html { margin-top: 71px !important; } * html body { margin-top: 71px !important; } #wpcom_request_access_iframe { display: block; height: 71px !important; } } #wpcom_request_access_iframe { border: 0px; height: 46px; position: fixed; top: 0; left: 0; width: 100%; min-width: 100%; z-index: 99999; background: #23282d; } ';

			var style = document.createElement( 'style' );
			style.type = 'text/css';
			style.id = 'wpcom_request_access_styles';
			style.appendChild( document.createTextNode( css ) );
			document.body.appendChild( style );

			document.body.appendChild( iframe );
		}

		if ( data.msg === 'DONE' ) {
			wpcom_remote_login_final_cleanup();
		}
	}
}, false );

// Inject the remote login iframe after the page has had a chance to load
// more critical resources
window.addEventListener( "DOMContentLoaded", function( e ) {
	var iframe = document.createElement( "iframe" );
	iframe.style.display = "none";
	iframe.setAttribute( "scrolling", "no" );
	iframe.setAttribute( "id", "wpcom_remote_login_key" );
	iframe.src = "https://r-login.wordpress.com/remote-login.php"
		+ "?wpcom_remote_login=key"
		+ "&origin=aHR0cHM6Ly9pbnRlbGxpZ2VudG1lYXN1cmVtZW50Lm5ldA%3D%3D"
		+ "&wpcomid=63261"
		+ "&time=" + Math.floor( Date.now() / 1000 );
	document.body.appendChild( iframe );
}, false );
</script>
<link rel='dns-prefetch' href='//s0.wp.com' />
<link rel='dns-prefetch' href='//widgets.wp.com' />
<link rel="alternate" type="application/rss+xml" title="intelligent measurement &raquo; Feed" href="https://intelligentmeasurement.net/feed/" />
<link rel="alternate" type="application/rss+xml" title="intelligent measurement &raquo; Comments Feed" href="https://intelligentmeasurement.net/comments/feed/" />
<link rel="alternate" type="application/rss+xml" title="intelligent measurement &raquo; Barcelona Principles 3.0 &#8211; communications&nbsp;evaluation Comments Feed" href="https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/feed/" />
	<script type="text/javascript">
		/* <![CDATA[ */
		function addLoadEvent(func) {
			var oldonload = window.onload;
			if (typeof window.onload != 'function') {
				window.onload = func;
			} else {
				window.onload = function () {
					oldonload();
					func();
				}
			}
		}
		/* ]]> */
	</script>
	<link crossorigin='anonymous' rel='stylesheet' id='all-css-0-1' href='/_static/??-eJx9jFsKwkAMAC/kGitK9UM8S7uNmjabDSZBvb0PBEHEz4GZgYumXMVRHEok5TiSGIzo2uXpzWAhUOoQjAZn5M5xSFrNv2iezWbw+8g04ef7on86yYGE/AZ+wvIINXroma7PZl92Tdsuts16tVmOdzvVSDw=&cssminify=yes' type='text/css' media='all' />
<style id="wp-emoji-styles-inline-css">

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
<style id="wp-block-library-inline-css">
:root{--wp-block-synced-color:#7a00df;--wp-block-synced-color--rgb:122,0,223;--wp-bound-block-color:var(--wp-block-synced-color);--wp-editor-canvas-background:#ddd;--wp-admin-theme-color:#007cba;--wp-admin-theme-color--rgb:0,124,186;--wp-admin-theme-color-darker-10:#006ba1;--wp-admin-theme-color-darker-10--rgb:0,107,160.5;--wp-admin-theme-color-darker-20:#005a87;--wp-admin-theme-color-darker-20--rgb:0,90,135;--wp-admin-border-width-focus:2px}@media (min-resolution:192dpi){:root{--wp-admin-border-width-focus:1.5px}}.wp-element-button{cursor:pointer}:root .has-very-light-gray-background-color{background-color:#eee}:root .has-very-dark-gray-background-color{background-color:#313131}:root .has-very-light-gray-color{color:#eee}:root .has-very-dark-gray-color{color:#313131}:root .has-vivid-green-cyan-to-vivid-cyan-blue-gradient-background{background:linear-gradient(135deg,#00d084,#0693e3)}:root .has-purple-crush-gradient-background{background:linear-gradient(135deg,#34e2e4,#4721fb 50%,#ab1dfe)}:root .has-hazy-dawn-gradient-background{background:linear-gradient(135deg,#faaca8,#dad0ec)}:root .has-subdued-olive-gradient-background{background:linear-gradient(135deg,#fafae1,#67a671)}:root .has-atomic-cream-gradient-background{background:linear-gradient(135deg,#fdd79a,#004a59)}:root .has-nightshade-gradient-background{background:linear-gradient(135deg,#330968,#31cdcf)}:root .has-midnight-gradient-background{background:linear-gradient(135deg,#020381,#2874fc)}:root{--wp--preset--font-size--normal:16px;--wp--preset--font-size--huge:42px}.has-regular-font-size{font-size:1em}.has-larger-font-size{font-size:2.625em}.has-normal-font-size{font-size:var(--wp--preset--font-size--normal)}.has-huge-font-size{font-size:var(--wp--preset--font-size--huge)}:root .has-text-align-center{text-align:center}:root .has-text-align-left{text-align:left}:root .has-text-align-right{text-align:right}.has-fit-text{white-space:nowrap!important}#end-resizable-editor-section{display:none}.aligncenter{clear:both}.items-justified-left{justify-content:flex-start}.items-justified-center{justify-content:center}.items-justified-right{justify-content:flex-end}.items-justified-space-between{justify-content:space-between}.screen-reader-text{word-wrap:normal!important;border:0;clip-path:inset(50%);height:1px;margin:-1px;overflow:hidden;padding:0;position:absolute;width:1px;word-break:normal!important}.screen-reader-text:focus{background-color:#ddd;clip-path:none;color:#444;display:block;font-size:1em;height:auto;left:5px;line-height:normal;padding:15px 23px 14px;text-decoration:none;top:5px;width:auto;z-index:100000}html :where(.has-border-color){border-style:solid}html :where([style*=border-color]){border-style:solid}html :where([style*=border-top-color]){border-top-style:solid}html :where([style*=border-right-color]){border-right-style:solid}html :where([style*=border-bottom-color]){border-bottom-style:solid}html :where([style*=border-left-color]){border-left-style:solid}html :where([style*=border-width]){border-style:solid}html :where([style*=border-top-width]){border-top-style:solid}html :where([style*=border-right-width]){border-right-style:solid}html :where([style*=border-bottom-width]){border-bottom-style:solid}html :where([style*=border-left-width]){border-left-style:solid}html :where(img[class*=wp-image-]){height:auto;max-width:100%}:where(figure){margin:0 0 1em}html :where(.is-position-sticky){--wp-admin--admin-bar--position-offset:var(--wp-admin--admin-bar--height,0px)}@media screen and (max-width:600px){html :where(.is-position-sticky){--wp-admin--admin-bar--position-offset:0px}}

/*# sourceURL=/wp-content/plugins/gutenberg-core/v23.1.0/build/styles/block-library/common.min.css */
</style>
<style id="wp-block-library-inline-css-extra">
.has-text-align-justify {
	text-align:justify;
}
.has-text-align-justify{text-align:justify;}
/*# sourceURL=wp-block-library-inline-css */
</style>

<style id="classic-theme-styles-inline-css">
.wp-block-button__link{background-color:#32373c;border-radius:9999px;box-shadow:none;color:#fff;font-size:1.125em;padding:calc(.667em + 2px) calc(1.333em + 2px);text-decoration:none}.wp-block-file__button{background:#32373c;color:#fff}.wp-block-accordion-heading{margin:0}.wp-block-accordion-heading__toggle{background-color:inherit!important;color:inherit!important}.wp-block-accordion-heading__toggle:not(:focus-visible){outline:none}.wp-block-accordion-heading__toggle:focus,.wp-block-accordion-heading__toggle:hover{background-color:inherit!important;border:none;box-shadow:none;color:inherit;padding:var(--wp--preset--spacing--20,1em) 0;text-decoration:none}.wp-block-accordion-heading__toggle:focus-visible{outline:auto;outline-offset:0}.wp-block-tab:not(.has-text-color){color:inherit!important}.wp-block-tab:not(.has-background){background-color:inherit!important}.wp-block-tab:focus,.wp-block-tab:hover{text-decoration:none}.wp-block-tab:focus-visible{outline:auto;outline-offset:0}
/*# sourceURL=/wp-content/plugins/gutenberg-core/v23.1.0/build/styles/block-library/classic.min.css */
</style>

<link crossorigin='anonymous' rel='stylesheet' id='all-css-8-1' href='/_static/??/wp-content/mu-plugins/core-compat/wp-mediaelement.css,/wp-content/mu-plugins/wpcom-bbpress-premium-themes.css?m=1432920480j&cssminify=yes' type='text/css' media='all' />
<style id="global-styles-inline-css">
:root{--wp--preset--aspect-ratio--square: 1;--wp--preset--aspect-ratio--4-3: 4/3;--wp--preset--aspect-ratio--3-4: 3/4;--wp--preset--aspect-ratio--3-2: 3/2;--wp--preset--aspect-ratio--2-3: 2/3;--wp--preset--aspect-ratio--16-9: 16/9;--wp--preset--aspect-ratio--9-16: 9/16;--wp--preset--color--black: #000000;--wp--preset--color--cyan-bluish-gray: #abb8c3;--wp--preset--color--white: #ffffff;--wp--preset--color--pale-pink: #f78da7;--wp--preset--color--vivid-red: #cf2e2e;--wp--preset--color--luminous-vivid-orange: #ff6900;--wp--preset--color--luminous-vivid-amber: #fcb900;--wp--preset--color--light-green-cyan: #7bdcb5;--wp--preset--color--vivid-green-cyan: #00d084;--wp--preset--color--pale-cyan-blue: #8ed1fc;--wp--preset--color--vivid-cyan-blue: #0693e3;--wp--preset--color--vivid-purple: #9b51e0;--wp--preset--gradient--vivid-cyan-blue-to-vivid-purple: linear-gradient(135deg,rgb(6,147,227) 0%,rgb(155,81,224) 100%);--wp--preset--gradient--light-green-cyan-to-vivid-green-cyan: linear-gradient(135deg,rgb(122,220,180) 0%,rgb(0,208,130) 100%);--wp--preset--gradient--luminous-vivid-amber-to-luminous-vivid-orange: linear-gradient(135deg,rgb(252,185,0) 0%,rgb(255,105,0) 100%);--wp--preset--gradient--luminous-vivid-orange-to-vivid-red: linear-gradient(135deg,rgb(255,105,0) 0%,rgb(207,46,46) 100%);--wp--preset--gradient--very-light-gray-to-cyan-bluish-gray: linear-gradient(135deg,rgb(238,238,238) 0%,rgb(169,184,195) 100%);--wp--preset--gradient--cool-to-warm-spectrum: linear-gradient(135deg,rgb(74,234,220) 0%,rgb(151,120,209) 20%,rgb(207,42,186) 40%,rgb(238,44,130) 60%,rgb(251,105,98) 80%,rgb(254,248,76) 100%);--wp--preset--gradient--blush-light-purple: linear-gradient(135deg,rgb(255,206,236) 0%,rgb(152,150,240) 100%);--wp--preset--gradient--blush-bordeaux: linear-gradient(135deg,rgb(254,205,165) 0%,rgb(254,45,45) 50%,rgb(107,0,62) 100%);--wp--preset--gradient--luminous-dusk: linear-gradient(135deg,rgb(255,203,112) 0%,rgb(199,81,192) 50%,rgb(65,88,208) 100%);--wp--preset--gradient--pale-ocean: linear-gradient(135deg,rgb(255,245,203) 0%,rgb(182,227,212) 50%,rgb(51,167,181) 100%);--wp--preset--gradient--electric-grass: linear-gradient(135deg,rgb(202,248,128) 0%,rgb(113,206,126) 100%);--wp--preset--gradient--midnight: linear-gradient(135deg,rgb(2,3,129) 0%,rgb(40,116,252) 100%);--wp--preset--font-size--small: 13px;--wp--preset--font-size--medium: 20px;--wp--preset--font-size--large: 36px;--wp--preset--font-size--x-large: 42px;--wp--preset--font-family--albert-sans: 'Albert Sans', sans-serif;--wp--preset--font-family--alegreya: Alegreya, serif;--wp--preset--font-family--arvo: Arvo, serif;--wp--preset--font-family--bodoni-moda: 'Bodoni Moda', serif;--wp--preset--font-family--bricolage-grotesque: 'Bricolage Grotesque', sans-serif;--wp--preset--font-family--cabin: Cabin, sans-serif;--wp--preset--font-family--chivo: Chivo, sans-serif;--wp--preset--font-family--commissioner: Commissioner, sans-serif;--wp--preset--font-family--cormorant: Cormorant, serif;--wp--preset--font-family--courier-prime: 'Courier Prime', monospace;--wp--preset--font-family--crimson-pro: 'Crimson Pro', serif;--wp--preset--font-family--dm-mono: 'DM Mono', monospace;--wp--preset--font-family--dm-sans: 'DM Sans', sans-serif;--wp--preset--font-family--dm-serif-display: 'DM Serif Display', serif;--wp--preset--font-family--domine: Domine, serif;--wp--preset--font-family--eb-garamond: 'EB Garamond', serif;--wp--preset--font-family--epilogue: Epilogue, sans-serif;--wp--preset--font-family--fahkwang: Fahkwang, sans-serif;--wp--preset--font-family--figtree: Figtree, sans-serif;--wp--preset--font-family--fira-sans: 'Fira Sans', sans-serif;--wp--preset--font-family--fjalla-one: 'Fjalla One', sans-serif;--wp--preset--font-family--fraunces: Fraunces, serif;--wp--preset--font-family--gabarito: Gabarito, system-ui;--wp--preset--font-family--ibm-plex-mono: 'IBM Plex Mono', monospace;--wp--preset--font-family--ibm-plex-sans: 'IBM Plex Sans', sans-serif;--wp--preset--font-family--ibarra-real-nova: 'Ibarra Real Nova', serif;--wp--preset--font-family--instrument-serif: 'Instrument Serif', serif;--wp--preset--font-family--inter: Inter, sans-serif;--wp--preset--font-family--josefin-sans: 'Josefin Sans', sans-serif;--wp--preset--font-family--jost: Jost, sans-serif;--wp--preset--font-family--libre-baskerville: 'Libre Baskerville', serif;--wp--preset--font-family--libre-franklin: 'Libre Franklin', sans-serif;--wp--preset--font-family--literata: Literata, serif;--wp--preset--font-family--lora: Lora, serif;--wp--preset--font-family--merriweather: Merriweather, serif;--wp--preset--font-family--montserrat: Montserrat, sans-serif;--wp--preset--font-family--newsreader: Newsreader, serif;--wp--preset--font-family--noto-sans-mono: 'Noto Sans Mono', sans-serif;--wp--preset--font-family--nunito: Nunito, sans-serif;--wp--preset--font-family--open-sans: 'Open Sans', sans-serif;--wp--preset--font-family--overpass: Overpass, sans-serif;--wp--preset--font-family--pt-serif: 'PT Serif', serif;--wp--preset--font-family--petrona: Petrona, serif;--wp--preset--font-family--piazzolla: Piazzolla, serif;--wp--preset--font-family--playfair-display: 'Playfair Display', serif;--wp--preset--font-family--plus-jakarta-sans: 'Plus Jakarta Sans', sans-serif;--wp--preset--font-family--poppins: Poppins, sans-serif;--wp--preset--font-family--raleway: Raleway, sans-serif;--wp--preset--font-family--roboto: Roboto, sans-serif;--wp--preset--font-family--roboto-slab: 'Roboto Slab', serif;--wp--preset--font-family--rubik: Rubik, sans-serif;--wp--preset--font-family--rufina: Rufina, serif;--wp--preset--font-family--sora: Sora, sans-serif;--wp--preset--font-family--source-sans-3: 'Source Sans 3', sans-serif;--wp--preset--font-family--source-serif-4: 'Source Serif 4', serif;--wp--preset--font-family--space-mono: 'Space Mono', monospace;--wp--preset--font-family--syne: Syne, sans-serif;--wp--preset--font-family--texturina: Texturina, serif;--wp--preset--font-family--urbanist: Urbanist, sans-serif;--wp--preset--font-family--work-sans: 'Work Sans', sans-serif;--wp--preset--spacing--20: 0.44rem;--wp--preset--spacing--30: 0.67rem;--wp--preset--spacing--40: 1rem;--wp--preset--spacing--50: 1.5rem;--wp--preset--spacing--60: 2.25rem;--wp--preset--spacing--70: 3.38rem;--wp--preset--spacing--80: 5.06rem;--wp--preset--shadow--natural: 6px 6px 9px rgba(0, 0, 0, 0.2);--wp--preset--shadow--deep: 12px 12px 50px rgba(0, 0, 0, 0.4);--wp--preset--shadow--sharp: 6px 6px 0px rgba(0, 0, 0, 0.2);--wp--preset--shadow--outlined: 6px 6px 0px -3px rgb(255, 255, 255), 6px 6px rgb(0, 0, 0);--wp--preset--shadow--crisp: 6px 6px 0px rgb(0, 0, 0);}.wp-block-button{--wp--preset--dimension--25: 25%;--wp--preset--dimension--50: 50%;--wp--preset--dimension--75: 75%;--wp--preset--dimension--100: 100%;}:where(body) { margin: 0; }:where(.is-layout-flex){gap: 0.5em;}:where(.is-layout-grid){gap: 0.5em;}body .is-layout-flex{display: flex;}.is-layout-flex{flex-wrap: wrap;align-items: center;}.is-layout-flex > :is(*, div){margin: 0;}body .is-layout-grid{display: grid;}.is-layout-grid > :is(*, div){margin: 0;}body{padding-top: 0px;padding-right: 0px;padding-bottom: 0px;padding-left: 0px;}:root :where(.wp-element-button, .wp-block-button__link){background-color: #32373c;border-width: 0;color: #fff;font-family: inherit;font-size: inherit;font-style: inherit;font-weight: inherit;letter-spacing: inherit;line-height: inherit;padding-top: calc(0.667em + 2px);padding-right: calc(1.333em + 2px);padding-bottom: calc(0.667em + 2px);padding-left: calc(1.333em + 2px);text-decoration: none;text-transform: inherit;}.has-black-color{color: var(--wp--preset--color--black) !important;}.has-cyan-bluish-gray-color{color: var(--wp--preset--color--cyan-bluish-gray) !important;}.has-white-color{color: var(--wp--preset--color--white) !important;}.has-pale-pink-color{color: var(--wp--preset--color--pale-pink) !important;}.has-vivid-red-color{color: var(--wp--preset--color--vivid-red) !important;}.has-luminous-vivid-orange-color{color: var(--wp--preset--color--luminous-vivid-orange) !important;}.has-luminous-vivid-amber-color{color: var(--wp--preset--color--luminous-vivid-amber) !important;}.has-light-green-cyan-color{color: var(--wp--preset--color--light-green-cyan) !important;}.has-vivid-green-cyan-color{color: var(--wp--preset--color--vivid-green-cyan) !important;}.has-pale-cyan-blue-color{color: var(--wp--preset--color--pale-cyan-blue) !important;}.has-vivid-cyan-blue-color{color: var(--wp--preset--color--vivid-cyan-blue) !important;}.has-vivid-purple-color{color: var(--wp--preset--color--vivid-purple) !important;}.has-black-background-color{background-color: var(--wp--preset--color--black) !important;}.has-cyan-bluish-gray-background-color{background-color: var(--wp--preset--color--cyan-bluish-gray) !important;}.has-white-background-color{background-color: var(--wp--preset--color--white) !important;}.has-pale-pink-background-color{background-color: var(--wp--preset--color--pale-pink) !important;}.has-vivid-red-background-color{background-color: var(--wp--preset--color--vivid-red) !important;}.has-luminous-vivid-orange-background-color{background-color: var(--wp--preset--color--luminous-vivid-orange) !important;}.has-luminous-vivid-amber-background-color{background-color: var(--wp--preset--color--luminous-vivid-amber) !important;}.has-light-green-cyan-background-color{background-color: var(--wp--preset--color--light-green-cyan) !important;}.has-vivid-green-cyan-background-color{background-color: var(--wp--preset--color--vivid-green-cyan) !important;}.has-pale-cyan-blue-background-color{background-color: var(--wp--preset--color--pale-cyan-blue) !important;}.has-vivid-cyan-blue-background-color{background-color: var(--wp--preset--color--vivid-cyan-blue) !important;}.has-vivid-purple-background-color{background-color: var(--wp--preset--color--vivid-purple) !important;}.has-black-border-color{border-color: var(--wp--preset--color--black) !important;}.has-cyan-bluish-gray-border-color{border-color: var(--wp--preset--color--cyan-bluish-gray) !important;}.has-white-border-color{border-color: var(--wp--preset--color--white) !important;}.has-pale-pink-border-color{border-color: var(--wp--preset--color--pale-pink) !important;}.has-vivid-red-border-color{border-color: var(--wp--preset--color--vivid-red) !important;}.has-luminous-vivid-orange-border-color{border-color: var(--wp--preset--color--luminous-vivid-orange) !important;}.has-luminous-vivid-amber-border-color{border-color: var(--wp--preset--color--luminous-vivid-amber) !important;}.has-light-green-cyan-border-color{border-color: var(--wp--preset--color--light-green-cyan) !important;}.has-vivid-green-cyan-border-color{border-color: var(--wp--preset--color--vivid-green-cyan) !important;}.has-pale-cyan-blue-border-color{border-color: var(--wp--preset--color--pale-cyan-blue) !important;}.has-vivid-cyan-blue-border-color{border-color: var(--wp--preset--color--vivid-cyan-blue) !important;}.has-vivid-purple-border-color{border-color: var(--wp--preset--color--vivid-purple) !important;}.has-vivid-cyan-blue-to-vivid-purple-gradient-background{background: var(--wp--preset--gradient--vivid-cyan-blue-to-vivid-purple) !important;}.has-light-green-cyan-to-vivid-green-cyan-gradient-background{background: var(--wp--preset--gradient--light-green-cyan-to-vivid-green-cyan) !important;}.has-luminous-vivid-amber-to-luminous-vivid-orange-gradient-background{background: var(--wp--preset--gradient--luminous-vivid-amber-to-luminous-vivid-orange) !important;}.has-luminous-vivid-orange-to-vivid-red-gradient-background{background: var(--wp--preset--gradient--luminous-vivid-orange-to-vivid-red) !important;}.has-very-light-gray-to-cyan-bluish-gray-gradient-background{background: var(--wp--preset--gradient--very-light-gray-to-cyan-bluish-gray) !important;}.has-cool-to-warm-spectrum-gradient-background{background: var(--wp--preset--gradient--cool-to-warm-spectrum) !important;}.has-blush-light-purple-gradient-background{background: var(--wp--preset--gradient--blush-light-purple) !important;}.has-blush-bordeaux-gradient-background{background: var(--wp--preset--gradient--blush-bordeaux) !important;}.has-luminous-dusk-gradient-background{background: var(--wp--preset--gradient--luminous-dusk) !important;}.has-pale-ocean-gradient-background{background: var(--wp--preset--gradient--pale-ocean) !important;}.has-electric-grass-gradient-background{background: var(--wp--preset--gradient--electric-grass) !important;}.has-midnight-gradient-background{background: var(--wp--preset--gradient--midnight) !important;}.has-small-font-size{font-size: var(--wp--preset--font-size--small) !important;}.has-medium-font-size{font-size: var(--wp--preset--font-size--medium) !important;}.has-large-font-size{font-size: var(--wp--preset--font-size--large) !important;}.has-x-large-font-size{font-size: var(--wp--preset--font-size--x-large) !important;}.has-albert-sans-font-family{font-family: var(--wp--preset--font-family--albert-sans) !important;}.has-alegreya-font-family{font-family: var(--wp--preset--font-family--alegreya) !important;}.has-arvo-font-family{font-family: var(--wp--preset--font-family--arvo) !important;}.has-bodoni-moda-font-family{font-family: var(--wp--preset--font-family--bodoni-moda) !important;}.has-bricolage-grotesque-font-family{font-family: var(--wp--preset--font-family--bricolage-grotesque) !important;}.has-cabin-font-family{font-family: var(--wp--preset--font-family--cabin) !important;}.has-chivo-font-family{font-family: var(--wp--preset--font-family--chivo) !important;}.has-commissioner-font-family{font-family: var(--wp--preset--font-family--commissioner) !important;}.has-cormorant-font-family{font-family: var(--wp--preset--font-family--cormorant) !important;}.has-courier-prime-font-family{font-family: var(--wp--preset--font-family--courier-prime) !important;}.has-crimson-pro-font-family{font-family: var(--wp--preset--font-family--crimson-pro) !important;}.has-dm-mono-font-family{font-family: var(--wp--preset--font-family--dm-mono) !important;}.has-dm-sans-font-family{font-family: var(--wp--preset--font-family--dm-sans) !important;}.has-dm-serif-display-font-family{font-family: var(--wp--preset--font-family--dm-serif-display) !important;}.has-domine-font-family{font-family: var(--wp--preset--font-family--domine) !important;}.has-eb-garamond-font-family{font-family: var(--wp--preset--font-family--eb-garamond) !important;}.has-epilogue-font-family{font-family: var(--wp--preset--font-family--epilogue) !important;}.has-fahkwang-font-family{font-family: var(--wp--preset--font-family--fahkwang) !important;}.has-figtree-font-family{font-family: var(--wp--preset--font-family--figtree) !important;}.has-fira-sans-font-family{font-family: var(--wp--preset--font-family--fira-sans) !important;}.has-fjalla-one-font-family{font-family: var(--wp--preset--font-family--fjalla-one) !important;}.has-fraunces-font-family{font-family: var(--wp--preset--font-family--fraunces) !important;}.has-gabarito-font-family{font-family: var(--wp--preset--font-family--gabarito) !important;}.has-ibm-plex-mono-font-family{font-family: var(--wp--preset--font-family--ibm-plex-mono) !important;}.has-ibm-plex-sans-font-family{font-family: var(--wp--preset--font-family--ibm-plex-sans) !important;}.has-ibarra-real-nova-font-family{font-family: var(--wp--preset--font-family--ibarra-real-nova) !important;}.has-instrument-serif-font-family{font-family: var(--wp--preset--font-family--instrument-serif) !important;}.has-inter-font-family{font-family: var(--wp--preset--font-family--inter) !important;}.has-josefin-sans-font-family{font-family: var(--wp--preset--font-family--josefin-sans) !important;}.has-jost-font-family{font-family: var(--wp--preset--font-family--jost) !important;}.has-libre-baskerville-font-family{font-family: var(--wp--preset--font-family--libre-baskerville) !important;}.has-libre-franklin-font-family{font-family: var(--wp--preset--font-family--libre-franklin) !important;}.has-literata-font-family{font-family: var(--wp--preset--font-family--literata) !important;}.has-lora-font-family{font-family: var(--wp--preset--font-family--lora) !important;}.has-merriweather-font-family{font-family: var(--wp--preset--font-family--merriweather) !important;}.has-montserrat-font-family{font-family: var(--wp--preset--font-family--montserrat) !important;}.has-newsreader-font-family{font-family: var(--wp--preset--font-family--newsreader) !important;}.has-noto-sans-mono-font-family{font-family: var(--wp--preset--font-family--noto-sans-mono) !important;}.has-nunito-font-family{font-family: var(--wp--preset--font-family--nunito) !important;}.has-open-sans-font-family{font-family: var(--wp--preset--font-family--open-sans) !important;}.has-overpass-font-family{font-family: var(--wp--preset--font-family--overpass) !important;}.has-pt-serif-font-family{font-family: var(--wp--preset--font-family--pt-serif) !important;}.has-petrona-font-family{font-family: var(--wp--preset--font-family--petrona) !important;}.has-piazzolla-font-family{font-family: var(--wp--preset--font-family--piazzolla) !important;}.has-playfair-display-font-family{font-family: var(--wp--preset--font-family--playfair-display) !important;}.has-plus-jakarta-sans-font-family{font-family: var(--wp--preset--font-family--plus-jakarta-sans) !important;}.has-poppins-font-family{font-family: var(--wp--preset--font-family--poppins) !important;}.has-raleway-font-family{font-family: var(--wp--preset--font-family--raleway) !important;}.has-roboto-font-family{font-family: var(--wp--preset--font-family--roboto) !important;}.has-roboto-slab-font-family{font-family: var(--wp--preset--font-family--roboto-slab) !important;}.has-rubik-font-family{font-family: var(--wp--preset--font-family--rubik) !important;}.has-rufina-font-family{font-family: var(--wp--preset--font-family--rufina) !important;}.has-sora-font-family{font-family: var(--wp--preset--font-family--sora) !important;}.has-source-sans-3-font-family{font-family: var(--wp--preset--font-family--source-sans-3) !important;}.has-source-serif-4-font-family{font-family: var(--wp--preset--font-family--source-serif-4) !important;}.has-space-mono-font-family{font-family: var(--wp--preset--font-family--space-mono) !important;}.has-syne-font-family{font-family: var(--wp--preset--font-family--syne) !important;}.has-texturina-font-family{font-family: var(--wp--preset--font-family--texturina) !important;}.has-urbanist-font-family{font-family: var(--wp--preset--font-family--urbanist) !important;}.has-work-sans-font-family{font-family: var(--wp--preset--font-family--work-sans) !important;}
/*# sourceURL=global-styles-inline-css */
</style>

<link crossorigin='anonymous' rel='stylesheet' id='all-css-10-1' href='/wp-content/mu-plugins/jetpack-mu-wpcom-plugin/sun/jetpack_vendor/automattic/jetpack-mu-wpcom/src/build/verbum-comments/verbum-comments.css?m=1777406400i&cssminify=yes' type='text/css' media='all' />
<link rel='stylesheet' id='verbum-gutenberg-css-css' href='https://widgets.wp.com/verbum-block-editor/block-editor.css?ver=1771405207' media='all' />
<link crossorigin='anonymous' rel='stylesheet' id='all-css-12-1' href='/_static/??-eJx9i0EOgzAMwD60LmIgJg6It9AqdEVpU5FUsN+v3OCymy3ZsGfjOCkmBf1gRIFcLFgKB4h+CZ9O5AGXKhaTqfiQBDa0xL6ih1pd9N/kkQ2xmzVwuolZaA7buU5xbN5d0w9t++rXH7qLO+8=&cssminify=yes' type='text/css' media='all' />
<link crossorigin='anonymous' rel='stylesheet' id='print-css-13-1' href='/wp-content/mu-plugins/global-print/global-print.css?m=1465851035i&cssminify=yes' type='text/css' media='print' />
<style id="jetpack-global-styles-frontend-style-inline-css">
:root { --font-headings: unset; --font-base: unset; --font-headings-default: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Oxygen-Sans,Ubuntu,Cantarell,"Helvetica Neue",sans-serif; --font-base-default: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Oxygen-Sans,Ubuntu,Cantarell,"Helvetica Neue",sans-serif;}
/*# sourceURL=jetpack-global-styles-frontend-style-inline-css */
</style>
<link crossorigin='anonymous' rel='stylesheet' id='all-css-16-1' href='/_static/??-eJyNjcsKAjEMRX/IGgaqgwvxU6TTlDZjmhTTIv69D9yIG3f3wOUcuDUXVXqSDnW4xiOTGKyptxAvHwYbAlVxcDKwEq4JA+L9PUnyNppt4G/RmSTCMogRTCMFdqxZ7Qt+lL2k+owXD5l1Cfw6nOpxmme/837aH9YHfvpKpw==&cssminify=yes' type='text/css' media='all' />
<script id="jetpack_related-posts-js-extra">
var related_posts_js_options = {"post_heading":"h4"};
//# sourceURL=jetpack_related-posts-js-extra
</script>
<script id="wpcom-actionbar-placeholder-js-extra">
var actionbardata = {"siteID":"63261","postID":"2103","siteURL":"https://intelligentmeasurement.net","xhrURL":"https://intelligentmeasurement.net/wp-admin/admin-ajax.php","nonce":"0d227c72f5","isLoggedIn":"","statusMessage":"","subsEmailDefault":"instantly","proxyScriptUrl":"https://s0.wp.com/wp-content/js/wpcom-proxy-request.js?m=1513050504i&amp;ver=20211021","shortlink":"https://wp.me/pgsl-xV","i18n":{"followedText":"New posts from this site will now appear in your \u003Ca href=\"https://wordpress.com/reader\"\u003EReader\u003C/a\u003E","foldBar":"Collapse this bar","unfoldBar":"Expand this bar","shortLinkCopied":"Shortlink copied to clipboard."}};
//# sourceURL=wpcom-actionbar-placeholder-js-extra
</script>
<script id="jetpack-mu-wpcom-settings-js-before">
var JETPACK_MU_WPCOM_SETTINGS = {"assetsUrl":"https://s0.wp.com/wp-content/mu-plugins/jetpack-mu-wpcom-plugin/sun/jetpack_vendor/automattic/jetpack-mu-wpcom/src/build/"};
//# sourceURL=jetpack-mu-wpcom-settings-js-before
</script>
<script crossorigin='anonymous' type='text/javascript'  src='/_static/??/wp-content/mu-plugins/jetpack-plugin/sun/_inc/build/related-posts/related-posts.min.js,/wp-content/js/rlt-proxy.js?m=1755011788j'></script>
<script id="rlt-proxy-js-after">
	rltInitialize( {"token":null,"iframeOrigins":["https:\/\/widgets.wp.com"]} );
//# sourceURL=rlt-proxy-js-after
</script>
<link rel="EditURI" type="application/rsd+xml" title="RSD" href="https://intelligentmeasurement.wordpress.com/xmlrpc.php?rsd" />
<meta name="generator" content="WordPress.com" />
<link rel="canonical" href="https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/" />
<link rel='shortlink' href='https://wp.me/pgsl-xV' />
<link rel="alternate" type="application/json+oembed" href="https://public-api.wordpress.com/oembed/?format=json&amp;url=https%3A%2F%2Fintelligentmeasurement.net%2F2020%2F07%2F16%2Fbarcelona-principles-3-0-communications-evaluation%2F&amp;for=wpcom-auto-discovery" /><link rel="alternate" type="application/xml+oembed" href="https://public-api.wordpress.com/oembed/?format=xml&amp;url=https%3A%2F%2Fintelligentmeasurement.net%2F2020%2F07%2F16%2Fbarcelona-principles-3-0-communications-evaluation%2F&amp;for=wpcom-auto-discovery" />
<!-- Jetpack Open Graph Tags -->
<meta property="og:type" content="article" />
<meta property="og:title" content="Barcelona Principles 3.0 &#8211; communications evaluation" />
<meta property="og:url" content="https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/" />
<meta property="og:description" content="The Barcelona Principles &#8211; which set out the main principles for communication evaluation &#8211;  have been updated &#8211; version 3.0. Following is an explanation provided by AMEC. 1. Sett…" />
<meta property="article:published_time" content="2020-07-16T07:09:31+00:00" />
<meta property="article:modified_time" content="2020-07-16T07:09:31+00:00" />
<meta property="og:site_name" content="intelligent measurement" />
<meta property="og:image" content="https://i0.wp.com/intelligentmeasurement.net/wp-content/uploads/2020/07/amec-barcelona-principles-3-high-res.png?fit=720%2C1200&#038;ssl=1" />
<meta property="og:image:width" content="720" />
<meta property="og:image:height" content="1200" />
<meta property="og:image:alt" content="" />
<meta property="og:locale" content="en_US" />
<meta property="article:publisher" content="https://www.facebook.com/WordPresscom" />
<meta name="twitter:creator" content="@glenn_oneil" />
<meta name="twitter:site" content="@glenn_oneil" />
<meta name="twitter:text:title" content="Barcelona Principles 3.0 &#8211; communications&nbsp;evaluation" />
<meta name="twitter:image" content="https://i0.wp.com/intelligentmeasurement.net/wp-content/uploads/2020/07/amec-barcelona-principles-3-high-res.png?fit=720%2C1200&#038;ssl=1&#038;w=640" />
<meta name="twitter:card" content="summary_large_image" />

<!-- End Jetpack Open Graph Tags -->
<link rel="shortcut icon" type="image/x-icon" href="https://s0.wp.com/i/favicon.ico?m=1713425267i" sizes="16x16 24x24 32x32 48x48" />
<link rel="icon" type="image/x-icon" href="https://s0.wp.com/i/favicon.ico?m=1713425267i" sizes="16x16 24x24 32x32 48x48" />
<link rel="apple-touch-icon" href="https://s0.wp.com/i/webclip.png?m=1713868326i" />
<link rel='openid.server' href='https://intelligentmeasurement.net/?openidserver=1' />
<link rel='openid.delegate' href='https://intelligentmeasurement.net/' />
<link rel="search" type="application/opensearchdescription+xml" href="https://intelligentmeasurement.net/osd.xml" title="intelligent measurement" />
<link rel="search" type="application/opensearchdescription+xml" href="https://s1.wp.com/opensearch.xml" title="WordPress.com" />
<meta name="description" content="The Barcelona Principles - which set out the main principles for communication evaluation -  have been updated - version 3.0. Following is an explanation provided by AMEC. 1. Setting goals is an absolute prerequisite to communications planning, measurement, and evaluation. The founding principle of SMART (specific, measurable, actionable, relevant, and time-bound) goals as a foundation&hellip;" />
<style type="text/css">
#header{
	background: url(https://s0.wp.com/wp-content/themes/pub/blix/images/spring_flavour/header_bg.jpg) no-repeat;
}
#header h1 a, #desc {
	color:#009193;
}
#desc {
	margin-right: 30px;
}
</style>
<link crossorigin='anonymous' rel='stylesheet' id='all-css-0-3' href='/_static/??-eJyVjt0KwjAMRl/ILRWGvRIfRbYsSLY2KU3DXt+JP+CdXp6Pw+GDrXSo0kgaZO9K8huLwUKtjLi+GMwFriwIU1JcDWzjQrVHswP8HMg6eyIDHKu6Ufo47+HP3vOQc5p3nAwrl8a6u1/UZ5ZH+JLPxxjjEE5DCMsdvZZdGw==&cssminify=yes' type='text/css' media='all' />

</head>

<body class="wp-singular post-template-default single single-post postid-2103 single-format-standard wp-theme-pubblix customizer-styles-applied jetpack-reblog-enabled">
<div id="container">

<div id="header">
	<h1><a href="https://intelligentmeasurement.net">intelligent measurement</a></h1>
</div>

<div id="navigation">

	<form action="https://intelligentmeasurement.net/" method="get">
		<fieldset>
			<input value="" name="s" id="s" />
			<input type="submit" value="Go!" id="searchbutton" name="searchbutton" />
		</fieldset>
	</form>

	<ul class="menu">
	<li><a href="https://intelligentmeasurement.net">Home</a></li>
	<li class="page_item page-item-4"><a href="https://intelligentmeasurement.net/contact/">Contact</a></li>
<li class="page_item page-item-2"><a href="https://intelligentmeasurement.net/about/">About</a></li>
</ul>

</div><!-- /navigation -->

<hr class="low" />

<div id="content">



	<div id="post-2103" class="entry single post-2103 post type-post status-publish format-standard hentry category-communication-evaluation category-pr-evaluation">

		<h2>Barcelona Principles 3.0 &#8211; communications&nbsp;evaluation</h2>

		<p class="info">
   		<em class="date"><a href="https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/" title="7:09 am" rel="bookmark">July 16, 2020 at 7:09 am</a></em>   							<em class="comments-popup">
			<a href="https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/#respond" class="commentlink" >Leave a comment</a>			</em>
		   		   		</p>

		<p><img data-attachment-id="2104" data-permalink="https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/amec-barcelona-principles-3-high-res/" data-orig-file="https://intelligentmeasurement.net/wp-content/uploads/2020/07/amec-barcelona-principles-3-high-res.png" data-orig-size="1799,3000" data-comments-opened="1" data-image-meta="{&quot;aperture&quot;:&quot;0&quot;,&quot;credit&quot;:&quot;&quot;,&quot;camera&quot;:&quot;&quot;,&quot;caption&quot;:&quot;&quot;,&quot;created_timestamp&quot;:&quot;0&quot;,&quot;copyright&quot;:&quot;&quot;,&quot;focal_length&quot;:&quot;0&quot;,&quot;iso&quot;:&quot;0&quot;,&quot;shutter_speed&quot;:&quot;0&quot;,&quot;title&quot;:&quot;&quot;,&quot;orientation&quot;:&quot;0&quot;}" data-image-title="AMEC-Barcelona-Principles-3-High-Res" data-image-description="" data-image-caption="" data-large-file="https://intelligentmeasurement.net/wp-content/uploads/2020/07/amec-barcelona-principles-3-high-res.png?w=455" class="  wp-image-2104 alignleft" src="https://intelligentmeasurement.net/wp-content/uploads/2020/07/amec-barcelona-principles-3-high-res.png?w=419&#038;h=699" alt="AMEC-Barcelona-Principles-3-High-Res" width="419" height="699" srcset="https://intelligentmeasurement.net/wp-content/uploads/2020/07/amec-barcelona-principles-3-high-res.png?w=419&amp;h=699 419w, https://intelligentmeasurement.net/wp-content/uploads/2020/07/amec-barcelona-principles-3-high-res.png?w=838&amp;h=1397 838w, https://intelligentmeasurement.net/wp-content/uploads/2020/07/amec-barcelona-principles-3-high-res.png?w=90&amp;h=150 90w, https://intelligentmeasurement.net/wp-content/uploads/2020/07/amec-barcelona-principles-3-high-res.png?w=180&amp;h=300 180w, https://intelligentmeasurement.net/wp-content/uploads/2020/07/amec-barcelona-principles-3-high-res.png?w=768&amp;h=1281 768w, https://intelligentmeasurement.net/wp-content/uploads/2020/07/amec-barcelona-principles-3-high-res.png?w=614&amp;h=1024 614w" sizes="(max-width: 419px) 100vw, 419px" /></p>
<p>The Barcelona Principles &#8211; which set out the main principles for communication evaluation &#8211;  have been updated &#8211; version 3.0.</p>
<p>Following is an explanation provided by <a href="https://amecorg.com/resources/barcelona-principles-3-0/" target="_blank" rel="noopener">AMEC</a>.</p>
<p><strong>1. Setting goals is an absolute prerequisite to communications planning, measurement, and evaluation.</strong><br />
The founding principle of SMART (specific, measurable, actionable, relevant, and time-bound) goals as a foundation for communications planning has been promoted to an essential prerequisite. It pushes measurement and evaluation as a core component of the planning process, articulating target outcomes and how progress towards these will be assessed.</p>
<p><strong>2. Measurement and evaluation should identify outputs, outcomes, and potential impact.</strong><br />
Previously, the Principles recommended measuring outcomes, rather than simply counting outputs. The updated principles extend this to consider longer term impact of communications strategy.</p>
<p><strong>3. Outcomes and impact should be identified for stakeholders, society, and the organization.</strong><br />
From the original focus on business metrics, such as sales and revenue, the 2020 update embraces a more holistic view of performance. It allows the model to be more inclusive of a broader range of organisations and communications roles that are not necessarily profit-driven.</p>
<p><strong>4. Communication measurement and evaluation should include both qualitative and quantitative analysis.</strong><br />
“To understand the full impact of your work, it is crucial that you use the full suite of methods to measure those outcomes,” summarised Ben Levine AMEC Board Director in describing the evolution of this principle to not just quantify but also understand how messages are being received, believed and interpreted.</p>
<p><strong>5. AVEs are not the value of communication.</strong><br />
The message remains consistent and clear; “we continue to believe that AVEs do not demonstrate the value of our work.” It is important that communications measurement and evaluation employs a richer, more nuanced, and multi-faceted approach to understand the impact of communications.</p>
<p><strong>6.Holistic communication measurement and evaluation includes all relevant online and offline channels.</strong><br />
Our founding principle that social media can and should be measured is so obvious today. The 2020 iteration reflects the game-changing shift in social communications’ capabilities, opportunities, and influence, such that all relevant online and offline channels should be measured and evaluated equally. The AMEC measurement framework promotes clarity across earned, owned, shared, and paid channels to ensure consistency in approach towards a common goal.</p>
<p><strong>7. Communication measurement and evaluation are rooted in integrity and transparency to drive learning and insights.</strong><br />
Sound, consistent, and sustained measurement calls for integrity and transparency in recognition of today’s attention to data privacy and stewardship as organisations comply with new regulations, such as GDPR. This is also a statement that measurement isn’t simply about data collection and tracking, but about learning from evaluation and applying insight back into communications planning. It recognises the need to be transparent about the context in which programmes are run and being aware of any bias that may exist in the tools, methodologies and interpretations applied.</p>
<div id="jp-post-flair" class="sharedaddy sd-like-enabled sd-sharing-enabled"><div class="sharedaddy sd-sharing-enabled"><div class="robots-nocontent sd-block sd-social sd-social-icon-text sd-sharing"><h3 class="sd-title">Share this:</h3><div class="sd-content"><ul><li class="share-twitter"><a rel="nofollow noopener noreferrer"
				data-shared="sharing-twitter-2103"
				class="share-twitter sd-button share-icon"
				href="https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/?share=twitter"
				target="_blank"
				aria-labelledby="sharing-twitter-2103"
				>
				<span id="sharing-twitter-2103" hidden>Share on X (Opens in new window)</span>
				<span>X</span>
			</a></li><li class="share-facebook"><a rel="nofollow noopener noreferrer"
				data-shared="sharing-facebook-2103"
				class="share-facebook sd-button share-icon"
				href="https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/?share=facebook"
				target="_blank"
				aria-labelledby="sharing-facebook-2103"
				>
				<span id="sharing-facebook-2103" hidden>Share on Facebook (Opens in new window)</span>
				<span>Facebook</span>
			</a></li><li class="share-email"><a rel="nofollow noopener noreferrer"
				data-shared="sharing-email-2103"
				class="share-email sd-button share-icon"
				href="mailto:?subject=%5BShared%20Post%5D%20Barcelona%20Principles%203.0%20-%20communications%20evaluation&#038;body=https%3A%2F%2Fintelligentmeasurement.net%2F2020%2F07%2F16%2Fbarcelona-principles-3-0-communications-evaluation%2F&#038;share=email"
				target="_blank"
				aria-labelledby="sharing-email-2103"
				data-email-share-error-title="Do you have email set up?" data-email-share-error-text="If you&#039;re having problems sharing via email, you might not have email set up for your browser. You may need to create a new email yourself." data-email-share-nonce="90b783e318" data-email-share-track-url="https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/?share=email">
				<span id="sharing-email-2103" hidden>Email a link to a friend (Opens in new window)</span>
				<span>Email</span>
			</a></li><li class="share-print"><a rel="nofollow noopener noreferrer"
				data-shared="sharing-print-2103"
				class="share-print sd-button share-icon"
				href="https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/#print?share=print"
				target="_blank"
				aria-labelledby="sharing-print-2103"
				>
				<span id="sharing-print-2103" hidden>Print (Opens in new window)</span>
				<span>Print</span>
			</a></li><li class="share-linkedin"><a rel="nofollow noopener noreferrer"
				data-shared="sharing-linkedin-2103"
				class="share-linkedin sd-button share-icon"
				href="https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/?share=linkedin"
				target="_blank"
				aria-labelledby="sharing-linkedin-2103"
				>
				<span id="sharing-linkedin-2103" hidden>Share on LinkedIn (Opens in new window)</span>
				<span>LinkedIn</span>
			</a></li><li class="share-end"></li></ul></div></div></div><div class='sharedaddy sd-block sd-like jetpack-likes-widget-wrapper jetpack-likes-widget-unloaded' id='like-post-wrapper-63261-2103-6a005d8e4b65f' data-src='//widgets.wp.com/likes/index.html?ver=20260510#blog_id=63261&amp;post_id=2103&amp;origin=intelligentmeasurement.wordpress.com&amp;obj_id=63261-2103-6a005d8e4b65f&amp;domain=intelligentmeasurement.net' data-name='like-post-frame-63261-2103-6a005d8e4b65f' data-title='Like or Reblog'><div class='likes-widget-placeholder post-likes-widget-placeholder' style='height: 55px;'><span class='button'><span>Like</span></span> <span class='loading'>Loading...</span></div><span class='sd-text-color'></span><a class='sd-link-color'></a></div>
<div id='jp-relatedposts' class='jp-relatedposts' >
	<h3 class="jp-relatedposts-headline"><em>Related</em></h3>
</div></div>		
		<p id="filedunder">
			Entry filed under: <a href="https://intelligentmeasurement.net/category/communication-evaluation/" rel="category tag">Communication evaluation</a>, <a href="https://intelligentmeasurement.net/category/pr-evaluation/" rel="category tag">PR evaluation</a>.		</p>

   </div>

	<p id="entrynavigation">
		<span class="previous"><a href="https://intelligentmeasurement.net/2020/06/09/amec-global-virtual-summit-on-communication-measurement-2020/">AMEC global virtual summit on communication measurement&nbsp;2020</a></span>		<span class="next"><a href="https://intelligentmeasurement.net/2020/09/01/implications-of-covid-19-on-evaluation/">Implications of COVID-19 on&nbsp;evaluation</a></span>	</p>



<!-- comments ................................. -->
<div id="comments">


	<div id="respond" class="comment-respond">
		<h3 id="reply-title" class="comment-reply-title">Leave a comment <small><a rel="nofollow" id="cancel-comment-reply-link" href="/2020/07/16/barcelona-principles-3-0-communications-evaluation/#respond" style="display:none;">Cancel reply</a></small></h3><form action="https://intelligentmeasurement.net/wp-comments-post.php" method="post" id="commentform" class="comment-form">


<div class="comment-form__verbum transparent"></div><div class="verbum-form-meta"><input type='hidden' name='comment_post_ID' value='2103' id='comment_post_ID' />
<input type='hidden' name='comment_parent' id='comment_parent' value='0' />

			<input type="hidden" name="highlander_comment_nonce" id="highlander_comment_nonce" value="29565ae90c" />
			<input type="hidden" name="verbum_show_subscription_modal" value="" /></div><p style="display: none;"><input type="hidden" id="akismet_comment_nonce" name="akismet_comment_nonce" value="215b9c1060" /></p><p style="display: none !important;" class="akismet-fields-container" data-prefix="ak_"><label>&#916;<textarea name="ak_hp_textarea" cols="45" rows="8" maxlength="100"></textarea></label><input type="hidden" id="ak_js_1" name="ak_js" value="89"/><script>
document.getElementById( "ak_js_1" ).setAttribute( "value", ( new Date() ).getTime() );
</script>
</p></form>	</div><!-- #respond -->
	
	<p><a href="https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/trackback/">Trackback this post</a> &nbsp;|&nbsp; <a href="https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/feed/">Subscribe to the comments via RSS Feed</a></p>

</div> <!-- /comments -->

</div> <!-- /content -->

<hr class="low" />

<div id="subcontent">


<div id="blog_subscription-3" class="widget widget_blog_subscription jetpack_subscription_widget"><h2><em><label for="subscribe-field">Email Subscription</label></em></h2>

			<div class="wp-block-jetpack-subscriptions__container">
			<form
				action="https://subscribe.wordpress.com"
				method="post"
				accept-charset="utf-8"
				data-blog="63261"
				data-post_access_level="everybody"
				id="subscribe-blog"
			>
				<p>Enter your email address to subscribe to this blog and receive notifications of new posts by email.</p>
				<p id="subscribe-email">
					<label
						id="subscribe-field-label"
						for="subscribe-field"
						class="screen-reader-text"
					>
						Email Address:					</label>

					<input
							type="email"
							name="email"
							autocomplete="email"
							
							style="width: 95%; padding: 1px 10px"
							placeholder="Email Address"
							value=""
							id="subscribe-field"
							required
						/>				</p>

				<p id="subscribe-submit"
									>
					<input type="hidden" name="action" value="subscribe"/>
					<input type="hidden" name="blog_id" value="63261"/>
					<input type="hidden" name="source" value="https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/"/>
					<input type="hidden" name="sub-type" value="widget"/>
					<input type="hidden" name="redirect_fragment" value="subscribe-blog"/>
					<input type="hidden" id="_wpnonce" name="_wpnonce" value="0fe9000e64" />					<button type="submit"
													class="wp-block-button__link"
																	>
						Sign me up!					</button>
				</p>
			</form>
							<div class="wp-block-jetpack-subscriptions__subscount">
					Join 2,666 other subscribers				</div>
						</div>
			
</div><div id="search-3" class="widget widget_search"><form role="search" method="get" id="searchform" class="searchform" action="https://intelligentmeasurement.net/">
				<div>
					<label class="screen-reader-text" for="s">Search for:</label>
					<input type="text" value="" name="s" id="s" />
					<input type="submit" id="searchsubmit" value="Search" />
				</div>
			</form></div>	<h2><em>Categories</em></h2>

	<ul class="categories">
		<li class="cat-item cat-item-26043440"><a href="https://intelligentmeasurement.net/category/advocacy-evaluation/">Advocacy evaluation</a>
</li>
	<li class="cat-item cat-item-2637883"><a href="https://intelligentmeasurement.net/category/campaign-evaluation/">Campaign evaluation</a>
</li>
	<li class="cat-item cat-item-218963379"><a href="https://intelligentmeasurement.net/category/communicating-evaluation-results/">communicating evaluation results</a>
</li>
	<li class="cat-item cat-item-3715087"><a href="https://intelligentmeasurement.net/category/communication-evaluation/">Communication evaluation</a>
</li>
	<li class="cat-item cat-item-58163344"><a href="https://intelligentmeasurement.net/category/conference-event-evaluation/">Conference / event evaluation</a>
</li>
	<li class="cat-item cat-item-9500241"><a href="https://intelligentmeasurement.net/category/development-evaluation/">Development evaluation</a>
</li>
	<li class="cat-item cat-item-7744515"><a href="https://intelligentmeasurement.net/category/evaluation-methodology/">Evaluation methodology</a>
</li>
	<li class="cat-item cat-item-9500256"><a href="https://intelligentmeasurement.net/category/evaluation-reporting/">Evaluation reporting</a>
</li>
	<li class="cat-item cat-item-9500541"><a href="https://intelligentmeasurement.net/category/evaluation-tools-surveys-interviews/">Evaluation tools (surveys, interviews..)</a>
</li>
	<li class="cat-item cat-item-98555381"><a href="https://intelligentmeasurement.net/category/evaluation-use-2/">Evaluation use</a>
</li>
	<li class="cat-item cat-item-54"><a href="https://intelligentmeasurement.net/category/general/">General</a>
</li>
	<li class="cat-item cat-item-9500249"><a href="https://intelligentmeasurement.net/category/media-analysis-monitoring/">Media analysis &amp; monitoring</a>
</li>
	<li class="cat-item cat-item-9500255"><a href="https://intelligentmeasurement.net/category/network-analysis-mapping/">Network analysis &amp; mapping</a>
</li>
	<li class="cat-item cat-item-38917"><a href="https://intelligentmeasurement.net/category/pr-evaluation/">PR evaluation</a>
</li>
	<li class="cat-item cat-item-7597"><a href="https://intelligentmeasurement.net/category/research-studies/">Research &#038; Studies</a>
</li>
	<li class="cat-item cat-item-4140277"><a href="https://intelligentmeasurement.net/category/social-media-monitoring/">Social media monitoring</a>
</li>
	<li class="cat-item cat-item-38918"><a href="https://intelligentmeasurement.net/category/training-evaluation/">Training evaluation</a>
</li>
	<li class="cat-item cat-item-15798"><a href="https://intelligentmeasurement.net/category/trainings-seminars-conferences/">Trainings, Seminars &#038; Conferences</a>
</li>
	<li class="cat-item cat-item-298493"><a href="https://intelligentmeasurement.net/category/web-metrics/">Web metrics</a>
</li>
	</ul>
	<h2><em>Feeds</em></h2>

	<ul class="feeds">
	<li><a href="https://intelligentmeasurement.net/feed/ ">Entries (RSS)</a></li>
	<li><a href="https://intelligentmeasurement.net/comments/feed/ ">Comments (RSS)</a></li>
	</ul>
<div id="linkcat-1356" class="widget widget_links"><h2><em>Blogroll</em></h2>
	<ul class='xoxo blogroll'>
<li><a href="http://aea365.org/blog/" title="daily blogging from the American Evaluation Association">AEA365 Blog</a></li>
<li><a href="http://betterevaluation.org/" title="M&amp;E resources">BetterEvaluation</a></li>
<li><a href="http://www.mande.co.uk/" title="Monitoring &#038; Evaluation News &#8211; focus on development &#038; NGOs">Monitoring &#038; Evaluation News</a></li>
<li><a href="http://onthinktanks.org/" title="Interesting insights into how research and advocacy influences policy">On Think Tanks</a></li>
<li><a href="http://www.owlre.com" title="Blog author&#8217;s company &#8211; specialises in evaluation activities in the communications, training/events and development fields">Owl RE</a></li>

	</ul>
</div>

</div> <!-- /subcontent -->


<hr class="low" />

<div id="footer">

	<p><a href="https://wordpress.com/?ref=footer_blog" rel="nofollow">Blog at WordPress.com.</a> </p>

</div> <!-- /footer -->

</div> <!-- /container -->

<!--  -->
<script type="speculationrules">
{"prefetch":[{"source":"document","where":{"and":[{"href_matches":"/*"},{"not":{"href_matches":["/wp-*.php","/wp-admin/*","/files/*","/wp-content/*","/wp-content/plugins/*","/wp-content/themes/pub/blix/*","/*\\?(.+)"]}},{"not":{"selector_matches":"a[rel~=\"nofollow\"]"}},{"not":{"selector_matches":".no-prefetch, .no-prefetch a"}}]},"eagerness":"conservative"}]}
</script>
<script id="grofiles-cards-js" src="//0.gravatar.com/js/hovercards/hovercards.min.js?ver=202619d1133547e8a1f2801609f51cef12522abe7f6647d869e9fbc974fe356a37e613"></script>
<script id="wpgroho-js-extra">
var WPGroHo = {"my_hash":""};
//# sourceURL=wpgroho-js-extra
</script>
<script crossorigin='anonymous' type='text/javascript'  src='/wp-content/mu-plugins/gravatar-hovercards/wpgroho.js?m=1610363240i'></script>

	<script>
		// Initialize and attach hovercards to all gravatars
		( function() {
			function init() {
				if ( typeof Gravatar === 'undefined' ) {
					return;
				}

				if ( typeof Gravatar.init !== 'function' ) {
					return;
				}

				Gravatar.profile_cb = function ( hash, id ) {
					WPGroHo.syncProfileData( hash, id );
				};

				Gravatar.my_hash = WPGroHo.my_hash;
				Gravatar.init(
					'body',
					'#wp-admin-bar-my-account',
					{
						i18n: {
							'Edit your profile →': 'Edit your profile →',
							'View profile →': 'View profile →',
							'Contact': 'Contact',
							'Send money': 'Send money',
							'Sorry, we are unable to load this Gravatar profile.': 'Sorry, we are unable to load this Gravatar profile.',
							'Gravatar not found.': 'Gravatar not found.',
							'Too Many Requests.': 'Too Many Requests.',
							'Internal Server Error.': 'Internal Server Error.',
							'Is this you?': 'Is this you?',
							'Claim your free profile.': 'Claim your free profile.',
							'Email': 'Email',
							'Home Phone': 'Home Phone',
							'Work Phone': 'Work Phone',
							'Cell Phone': 'Cell Phone',
							'Contact Form': 'Contact Form',
							'Calendar': 'Calendar',
						},
					}
				);
			}

			if ( document.readyState !== 'loading' ) {
				init();
			} else {
				document.addEventListener( 'DOMContentLoaded', init );
			}
		} )();
	</script>

		<div style="display:none">
	</div>
		<div id="actionbar" dir="ltr" style="display: none;"
			class="actnbr-pub-blix actnbr-has-follow actnbr-has-actions">
		<ul>
								<li class="actnbr-btn actnbr-hidden">
						<a class="actnbr-action actnbr-actn-comment" href="https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/#respond">
							<svg class="gridicon gridicons-comment" height="20" width="20" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><g><path d="M12 16l-5 5v-5H5c-1.1 0-2-.9-2-2V5c0-1.1.9-2 2-2h14c1.1 0 2 .9 2 2v9c0 1.1-.9 2-2 2h-7z"/></g></svg>							<span>Comment						</span>
						</a>
					</li>
									<li class="actnbr-btn actnbr-hidden">
						<a class="actnbr-action actnbr-actn-reblog" href="">
							<svg class="gridicon gridicons-reblog" height="20" width="20" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><g><path d="M22.086 9.914L20 7.828V18c0 1.105-.895 2-2 2h-7v-2h7V7.828l-2.086 2.086L14.5 8.5 19 4l4.5 4.5-1.414 1.414zM6 16.172V6h7V4H6c-1.105 0-2 .895-2 2v10.172l-2.086-2.086L.5 15.5 5 20l4.5-4.5-1.414-1.414L6 16.172z"/></g></svg><span>Reblog</span>
						</a>
					</li>
									<li class="actnbr-btn actnbr-hidden">
								<a class="actnbr-action actnbr-actn-follow " href="">
			<svg class="gridicon" height="20" width="20" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path clip-rule="evenodd" d="m4 4.5h12v6.5h1.5v-6.5-1.5h-1.5-12-1.5v1.5 10.5c0 1.1046.89543 2 2 2h7v-1.5h-7c-.27614 0-.5-.2239-.5-.5zm10.5 2h-9v1.5h9zm-5 3h-4v1.5h4zm3.5 1.5h-1v1h1zm-1-1.5h-1.5v1.5 1 1.5h1.5 1 1.5v-1.5-1-1.5h-1.5zm-2.5 2.5h-4v1.5h4zm6.5 1.25h1.5v2.25h2.25v1.5h-2.25v2.25h-1.5v-2.25h-2.25v-1.5h2.25z"  fill-rule="evenodd"></path></svg>
			<span>Subscribe</span>
		</a>
		<a class="actnbr-action actnbr-actn-following  no-display" href="">
			<svg class="gridicon" height="20" width="20" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path fill-rule="evenodd" clip-rule="evenodd" d="M16 4.5H4V15C4 15.2761 4.22386 15.5 4.5 15.5H11.5V17H4.5C3.39543 17 2.5 16.1046 2.5 15V4.5V3H4H16H17.5V4.5V12.5H16V4.5ZM5.5 6.5H14.5V8H5.5V6.5ZM5.5 9.5H9.5V11H5.5V9.5ZM12 11H13V12H12V11ZM10.5 9.5H12H13H14.5V11V12V13.5H13H12H10.5V12V11V9.5ZM5.5 12H9.5V13.5H5.5V12Z" fill="#008A20"></path><path class="following-icon-tick" d="M13.5 16L15.5 18L19 14.5" stroke="#008A20" stroke-width="1.5"></path></svg>
			<span>Subscribed</span>
		</a>
							<div class="actnbr-popover tip tip-top-left actnbr-notice" id="follow-bubble">
							<div class="tip-arrow"></div>
							<div class="tip-inner actnbr-follow-bubble">
															<ul>
											<li class="actnbr-sitename">
			<a href="https://intelligentmeasurement.net">
				<img loading='lazy' alt='' src='https://s0.wp.com/i/logo/wpcom-gray-white.png?m=1479929237i' srcset='https://s0.wp.com/i/logo/wpcom-gray-white.png 1x' class='avatar avatar-50' height='50' width='50' />				intelligent measurement			</a>
		</li>
										<div class="actnbr-message no-display"></div>
									<form method="post" action="https://subscribe.wordpress.com" accept-charset="utf-8" style="display: none;">
																						<div class="actnbr-follow-count">Join 442 other subscribers</div>
																					<div>
										<input type="email" name="email" placeholder="Enter your email address" class="actnbr-email-field" aria-label="Enter your email address" />
										</div>
										<input type="hidden" name="action" value="subscribe" />
										<input type="hidden" name="blog_id" value="63261" />
										<input type="hidden" name="source" value="https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/" />
										<input type="hidden" name="sub-type" value="actionbar-follow" />
										<input type="hidden" id="_wpnonce" name="_wpnonce" value="0fe9000e64" />										<div class="actnbr-button-wrap">
											<button type="submit" value="Sign me up">
												Sign me up											</button>
										</div>
									</form>
									<li class="actnbr-login-nudge">
										<div>
											Already have a WordPress.com account? <a href="https://wordpress.com/log-in?redirect_to=https%3A%2F%2Fr-login.wordpress.com%2Fremote-login.php%3Faction%3Dlink%26back%3Dhttps%253A%252F%252Fintelligentmeasurement.net%252F2020%252F07%252F16%252Fbarcelona-principles-3-0-communications-evaluation%252F">Log in now.</a>										</div>
									</li>
								</ul>
															</div>
						</div>
					</li>
							<li class="actnbr-ellipsis actnbr-hidden">
				<svg class="gridicon gridicons-ellipsis" height="24" width="24" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><g><path d="M7 12c0 1.104-.896 2-2 2s-2-.896-2-2 .896-2 2-2 2 .896 2 2zm12-2c-1.104 0-2 .896-2 2s.896 2 2 2 2-.896 2-2-.896-2-2-2zm-7 0c-1.104 0-2 .896-2 2s.896 2 2 2 2-.896 2-2-.896-2-2-2z"/></g></svg>				<div class="actnbr-popover tip tip-top-left actnbr-more">
					<div class="tip-arrow"></div>
					<div class="tip-inner">
						<ul>
								<li class="actnbr-sitename">
			<a href="https://intelligentmeasurement.net">
				<img loading='lazy' alt='' src='https://s0.wp.com/i/logo/wpcom-gray-white.png?m=1479929237i' srcset='https://s0.wp.com/i/logo/wpcom-gray-white.png 1x' class='avatar avatar-50' height='50' width='50' />				intelligent measurement			</a>
		</li>
								<li class="actnbr-folded-follow">
										<a class="actnbr-action actnbr-actn-follow " href="">
			<svg class="gridicon" height="20" width="20" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path clip-rule="evenodd" d="m4 4.5h12v6.5h1.5v-6.5-1.5h-1.5-12-1.5v1.5 10.5c0 1.1046.89543 2 2 2h7v-1.5h-7c-.27614 0-.5-.2239-.5-.5zm10.5 2h-9v1.5h9zm-5 3h-4v1.5h4zm3.5 1.5h-1v1h1zm-1-1.5h-1.5v1.5 1 1.5h1.5 1 1.5v-1.5-1-1.5h-1.5zm-2.5 2.5h-4v1.5h4zm6.5 1.25h1.5v2.25h2.25v1.5h-2.25v2.25h-1.5v-2.25h-2.25v-1.5h2.25z"  fill-rule="evenodd"></path></svg>
			<span>Subscribe</span>
		</a>
		<a class="actnbr-action actnbr-actn-following  no-display" href="">
			<svg class="gridicon" height="20" width="20" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path fill-rule="evenodd" clip-rule="evenodd" d="M16 4.5H4V15C4 15.2761 4.22386 15.5 4.5 15.5H11.5V17H4.5C3.39543 17 2.5 16.1046 2.5 15V4.5V3H4H16H17.5V4.5V12.5H16V4.5ZM5.5 6.5H14.5V8H5.5V6.5ZM5.5 9.5H9.5V11H5.5V9.5ZM12 11H13V12H12V11ZM10.5 9.5H12H13H14.5V11V12V13.5H13H12H10.5V12V11V9.5ZM5.5 12H9.5V13.5H5.5V12Z" fill="#008A20"></path><path class="following-icon-tick" d="M13.5 16L15.5 18L19 14.5" stroke="#008A20" stroke-width="1.5"></path></svg>
			<span>Subscribed</span>
		</a>
								</li>
														<li class="actnbr-signup"><a href="https://wordpress.com/start/">Sign up</a></li>
							<li class="actnbr-login"><a href="https://wordpress.com/log-in?redirect_to=https%3A%2F%2Fr-login.wordpress.com%2Fremote-login.php%3Faction%3Dlink%26back%3Dhttps%253A%252F%252Fintelligentmeasurement.net%252F2020%252F07%252F16%252Fbarcelona-principles-3-0-communications-evaluation%252F">Log in</a></li>
																<li class="actnbr-shortlink">
										<a href="https://wp.me/pgsl-xV">
											<span class="actnbr-shortlink__text">Copy shortlink</span>
											<span class="actnbr-shortlink__icon"><svg class="gridicon gridicons-checkmark" height="16" width="16" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><g><path d="M9 19.414l-6.707-6.707 1.414-1.414L9 16.586 20.293 5.293l1.414 1.414"/></g></svg></span>
										</a>
									</li>
																<li class="flb-report">
									<a href="https://wordpress.com/abuse/?report_url=https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/" target="_blank" rel="noopener noreferrer">
										Report this content									</a>
								</li>
															<li class="actnbr-reader">
									<a href="https://wordpress.com/reader/blogs/63261/posts/2103">
										View post in Reader									</a>
								</li>
															<li class="actnbr-subs">
									<a href="https://subscribe.wordpress.com/">Manage subscriptions</a>
								</li>
																<li class="actnbr-fold"><a href="">Collapse this bar</a></li>
														</ul>
					</div>
				</div>
			</li>
		</ul>
	</div>
	
<script>
window.addEventListener( "DOMContentLoaded", function( event ) {
	var link = document.createElement( "link" );
	link.href = "/wp-content/mu-plugins/actionbar/actionbar.css?v=20250116";
	link.type = "text/css";
	link.rel = "stylesheet";
	document.head.appendChild( link );

	var script = document.createElement( "script" );
	script.src = "/wp-content/mu-plugins/actionbar/actionbar.js?v=20250204";
	document.body.appendChild( script );
} );
</script>

			<div id="jp-carousel-loading-overlay">
			<div id="jp-carousel-loading-wrapper">
				<span id="jp-carousel-library-loading"><svg class="jetpack-spinner" width="40" height="40" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false"><circle cx="50" cy="50" r="46" fill="none" stroke="#ddd" stroke-width="8"/><path d="M 50 4 A 46 46 0 0 1 96 50" fill="none" stroke="currentColor" stroke-width="8" stroke-linecap="round"><animateTransform attributeName="transform" type="rotate" dur="1.4s" from="0 50 50" to="360 50 50" repeatCount="indefinite"/></path></svg></span>
			</div>
		</div>
		<div class="jp-carousel-overlay" style="display: none;">

		<div class="jp-carousel-container">
			<!-- The Carousel Swiper -->
			<div
				class="jp-carousel-wrap swiper jp-carousel-swiper-container jp-carousel-transitions"
				itemscope
				itemtype="https://schema.org/ImageGallery">
				<div class="jp-carousel swiper-wrapper"></div>
				<div class="jp-swiper-button-prev swiper-button-prev">
					<svg width="25" height="24" viewBox="0 0 25 24" fill="none" xmlns="http://www.w3.org/2000/svg">
						<mask id="maskPrev" mask-type="alpha" maskUnits="userSpaceOnUse" x="8" y="6" width="9" height="12">
							<path d="M16.2072 16.59L11.6496 12L16.2072 7.41L14.8041 6L8.8335 12L14.8041 18L16.2072 16.59Z" fill="white"/>
						</mask>
						<g mask="url(#maskPrev)">
							<rect x="0.579102" width="23.8823" height="24" fill="#FFFFFF"/>
						</g>
					</svg>
				</div>
				<div class="jp-swiper-button-next swiper-button-next">
					<svg width="25" height="24" viewBox="0 0 25 24" fill="none" xmlns="http://www.w3.org/2000/svg">
						<mask id="maskNext" mask-type="alpha" maskUnits="userSpaceOnUse" x="8" y="6" width="8" height="12">
							<path d="M8.59814 16.59L13.1557 12L8.59814 7.41L10.0012 6L15.9718 12L10.0012 18L8.59814 16.59Z" fill="white"/>
						</mask>
						<g mask="url(#maskNext)">
							<rect x="0.34375" width="23.8822" height="24" fill="#FFFFFF"/>
						</g>
					</svg>
				</div>
			</div>
			<!-- The main close buton -->
			<div class="jp-carousel-close-hint">
				<svg width="25" height="24" viewBox="0 0 25 24" fill="none" xmlns="http://www.w3.org/2000/svg">
					<mask id="maskClose" mask-type="alpha" maskUnits="userSpaceOnUse" x="5" y="5" width="15" height="14">
						<path d="M19.3166 6.41L17.9135 5L12.3509 10.59L6.78834 5L5.38525 6.41L10.9478 12L5.38525 17.59L6.78834 19L12.3509 13.41L17.9135 19L19.3166 17.59L13.754 12L19.3166 6.41Z" fill="white"/>
					</mask>
					<g mask="url(#maskClose)">
						<rect x="0.409668" width="23.8823" height="24" fill="#FFFFFF"/>
					</g>
				</svg>
			</div>
			<!-- Image info, comments and meta -->
			<div class="jp-carousel-info">
				<div class="jp-carousel-info-footer">
					<div class="jp-carousel-pagination-container">
						<div class="jp-swiper-pagination swiper-pagination"></div>
						<div class="jp-carousel-pagination"></div>
					</div>
					<div class="jp-carousel-photo-title-container">
						<h2 class="jp-carousel-photo-caption"></h2>
					</div>
					<div class="jp-carousel-photo-icons-container">
						<a href="#" class="jp-carousel-icon-btn jp-carousel-icon-info" aria-label="Toggle photo metadata visibility">
							<span class="jp-carousel-icon">
								<svg width="25" height="24" viewBox="0 0 25 24" fill="none" xmlns="http://www.w3.org/2000/svg">
									<mask id="maskInfo" mask-type="alpha" maskUnits="userSpaceOnUse" x="2" y="2" width="21" height="20">
										<path fill-rule="evenodd" clip-rule="evenodd" d="M12.7537 2C7.26076 2 2.80273 6.48 2.80273 12C2.80273 17.52 7.26076 22 12.7537 22C18.2466 22 22.7046 17.52 22.7046 12C22.7046 6.48 18.2466 2 12.7537 2ZM11.7586 7V9H13.7488V7H11.7586ZM11.7586 11V17H13.7488V11H11.7586ZM4.79292 12C4.79292 16.41 8.36531 20 12.7537 20C17.142 20 20.7144 16.41 20.7144 12C20.7144 7.59 17.142 4 12.7537 4C8.36531 4 4.79292 7.59 4.79292 12Z" fill="white"/>
									</mask>
									<g mask="url(#maskInfo)">
										<rect x="0.8125" width="23.8823" height="24" fill="#FFFFFF"/>
									</g>
								</svg>
							</span>
						</a>
												<a href="#" class="jp-carousel-icon-btn jp-carousel-icon-comments" aria-label="Toggle photo comments visibility">
							<span class="jp-carousel-icon">
								<svg width="25" height="24" viewBox="0 0 25 24" fill="none" xmlns="http://www.w3.org/2000/svg">
									<mask id="maskComments" mask-type="alpha" maskUnits="userSpaceOnUse" x="2" y="2" width="21" height="20">
										<path fill-rule="evenodd" clip-rule="evenodd" d="M4.3271 2H20.2486C21.3432 2 22.2388 2.9 22.2388 4V16C22.2388 17.1 21.3432 18 20.2486 18H6.31729L2.33691 22V4C2.33691 2.9 3.2325 2 4.3271 2ZM6.31729 16H20.2486V4H4.3271V18L6.31729 16Z" fill="white"/>
									</mask>
									<g mask="url(#maskComments)">
										<rect x="0.34668" width="23.8823" height="24" fill="#FFFFFF"/>
									</g>
								</svg>

								<span class="jp-carousel-has-comments-indicator" aria-label="This image has comments."></span>
							</span>
						</a>
											</div>
				</div>
				<div class="jp-carousel-info-extra">
					<div class="jp-carousel-info-content-wrapper">
						<div class="jp-carousel-photo-title-container">
							<h2 class="jp-carousel-photo-title"></h2>
						</div>
						<div class="jp-carousel-comments-wrapper">
															<div id="jp-carousel-comments-loading">
									<span>Loading Comments...</span>
								</div>
								<div class="jp-carousel-comments"></div>
								<div id="jp-carousel-comment-form-container">
									<span id="jp-carousel-comment-form-spinner"><svg class="jetpack-spinner" width="20" height="20" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false"><circle cx="50" cy="50" r="46" fill="none" stroke="#ddd" stroke-width="8"/><path d="M 50 4 A 46 46 0 0 1 96 50" fill="none" stroke="currentColor" stroke-width="8" stroke-linecap="round"><animateTransform attributeName="transform" type="rotate" dur="1.4s" from="0 50 50" to="360 50 50" repeatCount="indefinite"/></path></svg></span>
									<div id="jp-carousel-comment-post-results"></div>
																														<form id="jp-carousel-comment-form">
												<label for="jp-carousel-comment-form-comment-field" class="screen-reader-text">Write a Comment...</label>
												<textarea
													name="comment"
													class="jp-carousel-comment-form-field jp-carousel-comment-form-textarea"
													id="jp-carousel-comment-form-comment-field"
													placeholder="Write a Comment..."
												></textarea>
												<div id="jp-carousel-comment-form-submit-and-info-wrapper">
													<div id="jp-carousel-comment-form-commenting-as">
																													<fieldset>
																<label for="jp-carousel-comment-form-email-field">Email (Required)</label>
																<input type="text" name="email" class="jp-carousel-comment-form-field jp-carousel-comment-form-text-field" id="jp-carousel-comment-form-email-field" />
															</fieldset>
															<fieldset>
																<label for="jp-carousel-comment-form-author-field">Name (Required)</label>
																<input type="text" name="author" class="jp-carousel-comment-form-field jp-carousel-comment-form-text-field" id="jp-carousel-comment-form-author-field" />
															</fieldset>
															<fieldset>
																<label for="jp-carousel-comment-form-url-field">Website</label>
																<input type="text" name="url" class="jp-carousel-comment-form-field jp-carousel-comment-form-text-field" id="jp-carousel-comment-form-url-field" />
															</fieldset>
																											</div>
													<input
														type="submit"
														name="submit"
														class="jp-carousel-comment-form-button"
														id="jp-carousel-comment-form-button-submit"
														value="Post Comment" />
												</div>
											</form>
																											</div>
													</div>
						<div class="jp-carousel-image-meta">
							<div class="jp-carousel-title-and-caption">
								<div class="jp-carousel-photo-info">
									<h3 class="jp-carousel-caption" itemprop="caption description"></h3>
								</div>

								<div class="jp-carousel-photo-description"></div>
							</div>
							<ul class="jp-carousel-image-exif" style="display: none;"></ul>
							<a class="jp-carousel-image-download" href="#" target="_blank" style="display: none;">
								<svg width="25" height="24" viewBox="0 0 25 24" fill="none" xmlns="http://www.w3.org/2000/svg">
									<mask id="mask0" mask-type="alpha" maskUnits="userSpaceOnUse" x="3" y="3" width="19" height="18">
										<path fill-rule="evenodd" clip-rule="evenodd" d="M5.84615 5V19H19.7775V12H21.7677V19C21.7677 20.1 20.8721 21 19.7775 21H5.84615C4.74159 21 3.85596 20.1 3.85596 19V5C3.85596 3.9 4.74159 3 5.84615 3H12.8118V5H5.84615ZM14.802 5V3H21.7677V10H19.7775V6.41L9.99569 16.24L8.59261 14.83L18.3744 5H14.802Z" fill="white"/>
									</mask>
									<g mask="url(#mask0)">
										<rect x="0.870605" width="23.8823" height="24" fill="#FFFFFF"/>
									</g>
								</svg>
								<span class="jp-carousel-download-text"></span>
							</a>
							<div class="jp-carousel-image-map" style="display: none;"></div>
						</div>
					</div>
				</div>
			</div>
		</div>

		</div>
		
	<script type="text/javascript">
		window.WPCOM_sharing_counts = {"https://intelligentmeasurement.net/2020/07/16/barcelona-principles-3-0-communications-evaluation/":2103};
	</script>
				
	<script type="text/javascript">
		(function () {
			var wpcom_reblog = {
				source: 'toolbar',

				toggle_reblog_box_flair: function (obj_id, post_id) {

					// Go to site selector. This will redirect to their blog if they only have one.
					const postEndpoint = `https://wordpress.com/post`;

					// Ideally we would use the permalink here, but fortunately this will be replaced with the 
					// post permalink in the editor.
					const originalURL = `${ document.location.href }?page_id=${ post_id }`; 
					
					const url =
						postEndpoint +
						'?url=' +
						encodeURIComponent( originalURL ) +
						'&is_post_share=true' +
						'&v=5';

					const redirect = function () {
						if (
							! window.open( url, '_blank' )
						) {
							location.href = url;
						}
					};

					if ( /Firefox/.test( navigator.userAgent ) ) {
						setTimeout( redirect, 0 );
					} else {
						redirect();
					}
				},
			};

			window.wpcom_reblog = wpcom_reblog;
		})();
	</script>
<script crossorigin='anonymous' type='text/javascript'  src='/wp-content/mu-plugins/likes/queuehandler.js?m=1741961244i'></script>
<script id="wp-hooks-js" crossorigin='anonymous' src="https://s0.wp.com/wp-content/plugins/gutenberg-core/v23.1.0/build/scripts/hooks/index.min.js?m=1778170586i&#038;ver=7496969728ca0f95732d"></script>
<script id="wp-i18n-js" crossorigin='anonymous' src="https://s0.wp.com/wp-content/plugins/gutenberg-core/v23.1.0/build/scripts/i18n/index.min.js?m=1778170586i&#038;ver=125448662852c5e18937"></script>
<script id="wp-i18n-js-after">
wp.i18n.setLocaleData( { 'text direction\u0004ltr': [ 'ltr' ] } );
//# sourceURL=wp-i18n-js-after
</script>
<script id="verbum-settings-js-before">
window.VerbumComments = {"Log in or provide your name and email to leave a reply.":"Log in or provide your name and email to leave a reply.","Log in or provide your name and email to leave a comment.":"Log in or provide your name and email to leave a comment.","Receive web and mobile notifications for posts on this site.":"Receive web and mobile notifications for posts on this site.","Name":"Name","Email (address never made public)":"Email (address never made public)","Website (optional)":"Website (optional)","Leave a reply. (log in optional)":"Leave a reply. (log in optional)","Leave a comment. (log in optional)":"Leave a comment. (log in optional)","Log in to leave a reply.":"Log in to leave a reply.","Log in to leave a comment.":"Log in to leave a comment.","Logged in via %s":"Logged in via %s","Log out":"Log out","Email":"Email","(Address never made public)":"(Address never made public)","Instantly":"Instantly","Daily":"Daily","Reply":"Reply","Comment":"Comment","WordPress":"WordPress","Weekly":"Weekly","Notify me of new posts":"Notify me of new posts","Email me new posts":"Email me new posts","Email me new comments":"Email me new comments","Cancel":"Cancel","Write a comment...":"Write a comment...","Write a reply...":"Write a reply...","Website":"Website","Optional":"Optional","We'll keep you in the loop!":"We'll keep you in the loop!","Loading your comment...":"Loading your comment...","Discover more from":"Discover more from intelligent measurement","Subscribe now to keep reading and get access to the full archive.":"Subscribe now to keep reading and get access to the full archive.","Continue reading":"Continue reading","Never miss a beat!":"Never miss a beat!","Interested in getting blog post updates? Simply click the button below to stay in the loop!":"Interested in getting blog post updates? Simply click the button below to stay in the loop!","Enter your email address":"Enter your email address","Subscribe":"Subscribe","Comment sent successfully":"Comment sent successfully","Save my name, email, and website in this browser for the next time I comment.":"Save my name, email, and website in this browser for the next time I comment.","hovercardi18n":{"Edit your profile \u2192":"Edit your profile \u2192","View profile \u2192":"View profile \u2192","Contact":"Contact","Send money":"Send money","Gravatar not found.":"Gravatar not found.","This profile is private.":"This profile is private.","Too Many Requests.":"Too Many Requests.","Internal Server Error.":"Internal Server Error.","Sorry, we are unable to load this Gravatar profile.":"Sorry, we are unable to load this Gravatar profile."},"siteId":63261,"postId":2103,"mustLogIn":false,"requireNameEmail":true,"commentRegistration":false,"connectURL":"https://intelligentmeasurement.wordpress.com/public.api/connect/?action=request\u0026domain=intelligentmeasurement.net","logoutURL":"https://intelligentmeasurement.wordpress.com/wp-login.php?action=logout\u0026_wpnonce=9959f1b468","homeURL":"https://intelligentmeasurement.net/","subscribeToBlog":true,"subscribeToComment":true,"isJetpackCommentsLoggedIn":false,"jetpackUsername":"","jetpackUserId":0,"jetpackSignature":"","jetpackAvatar":"https://0.gravatar.com/avatar/?s=96\u0026amp;d=identicon\u0026amp;r=G","enableBlocks":true,"enableSubscriptionModal":true,"currentLocale":"en","isJetpackComments":false,"allowedBlocks":["core/paragraph","core/list","core/code","core/list-item","core/quote","core/image","core/embed","core/quote","core/code"],"embedNonce":"97b07cb0c6","verbumBundleUrl":"/wp-content/mu-plugins/jetpack-mu-wpcom-plugin/sun/jetpack_vendor/automattic/jetpack-mu-wpcom/src/features/verbum-comments/dist/index.js","isRTL":false,"vbeCacheBuster":1771405207,"iframeUniqueId":0,"colorScheme":false}
//# sourceURL=verbum-settings-js-before
</script>
<script data-wp-strategy="defer" defer id="verbum-dynamic-loader-js" src="/wp-content/mu-plugins/jetpack-mu-wpcom-plugin/sun/jetpack_vendor/automattic/jetpack-mu-wpcom/src/build/verbum-comments/assets/dynamic-loader.js?m=1772522099i&#038;minify=false&#038;ver=8b5296f9195db92c42e4"></script>
<script id="jetpack-carousel-js-extra">
var jetpackSwiperLibraryPath = {"url":"/wp-content/mu-plugins/jetpack-plugin/sun/_inc/blocks/swiper.js"};
var jetpackCarouselStrings = {"widths":[370,700,1000,1200,1400,2000],"is_logged_in":"","lang":"en","ajaxurl":"https://intelligentmeasurement.net/wp-admin/admin-ajax.php","nonce":"188131027d","display_exif":"1","display_comments":"1","single_image_gallery":"1","single_image_gallery_media_file":"","background_color":"black","comment":"Comment","post_comment":"Post Comment","write_comment":"Write a Comment...","loading_comments":"Loading Comments...","image_label":"Open image in full-screen.","download_original":"View full size \u003Cspan class=\"photo-size\"\u003E{0}\u003Cspan class=\"photo-size-times\"\u003E\u00d7\u003C/span\u003E{1}\u003C/span\u003E","no_comment_text":"Please be sure to submit some text with your comment.","no_comment_email":"Please provide an email address to comment.","no_comment_author":"Please provide your name to comment.","comment_post_error":"Sorry, but there was an error posting your comment. Please try again later.","comment_approved":"Your comment was approved.","comment_unapproved":"Your comment is in moderation.","camera":"Camera","aperture":"Aperture","shutter_speed":"Shutter Speed","focal_length":"Focal Length","copyright":"Copyright","comment_registration":"0","require_name_email":"1","login_url":"https://intelligentmeasurement.wordpress.com/wp-login.php?redirect_to=https%3A%2F%2Fintelligentmeasurement.net%2F2020%2F07%2F16%2Fbarcelona-principles-3-0-communications-evaluation%2F","blog_id":"63261","meta_data":["camera","aperture","shutter_speed","focal_length","copyright"],"stats_query_args":"blog=63261&v=wpcom&tz=0&user_id=0&subd=intelligentmeasurement","is_public":"1"};
//# sourceURL=jetpack-carousel-js-extra
</script>
<script id="sharing-js-js-extra">
var sharing_js_options = {"lang":"en","counts":"1","is_stats_active":"1"};
//# sourceURL=sharing-js-js-extra
</script>
<script id="jetpack-script-data-js-before">
window.JetpackScriptData = {"site":{"host":"wpcom","is_wpcom_platform":true}};
//# sourceURL=jetpack-script-data-js-before
</script>
<script crossorigin='anonymous' type='text/javascript'  src='/_static/??-eJydUEkOwjAM/BBpWqFSLoinIJOYkjabYgfE74mgrRAHQNzsWTway2sUxiubNZIcSKrgHHoWCaO9Vc74aqCVLCIVPBdCuiyizb3xRY4cQY2iQNdYjBMhKfuZO1zQ65AkZA4OmI1aXECETPKYjdULSCqZyEIDww/BL3mHUmK6pSCFTGgX2Qx8qQOjIYcs1lX9PDcDp/QQ6w/eOeL5iLf17yZ0hoQatL49RuP7qcLe7Zquazfbpq2b4Q7KXKl5'></script>
<script id="sharing-js-js-after">
var windowOpen;
			( function () {
				function matches( el, sel ) {
					return !! (
						el.matches && el.matches( sel ) ||
						el.msMatchesSelector && el.msMatchesSelector( sel )
					);
				}

				document.body.addEventListener( 'click', function ( event ) {
					if ( ! event.target ) {
						return;
					}

					var el;
					if ( matches( event.target, 'a.share-twitter' ) ) {
						el = event.target;
					} else if ( event.target.parentNode && matches( event.target.parentNode, 'a.share-twitter' ) ) {
						el = event.target.parentNode;
					}

					if ( el ) {
						event.preventDefault();

						// If there's another sharing window open, close it.
						if ( typeof windowOpen !== 'undefined' ) {
							windowOpen.close();
						}
						windowOpen = window.open( el.getAttribute( 'href' ), 'wpcomtwitter', 'menubar=1,resizable=1,width=600,height=350' );
						return false;
					}
				} );
			} )();
var windowOpen;
			( function () {
				function matches( el, sel ) {
					return !! (
						el.matches && el.matches( sel ) ||
						el.msMatchesSelector && el.msMatchesSelector( sel )
					);
				}

				document.body.addEventListener( 'click', function ( event ) {
					if ( ! event.target ) {
						return;
					}

					var el;
					if ( matches( event.target, 'a.share-facebook' ) ) {
						el = event.target;
					} else if ( event.target.parentNode && matches( event.target.parentNode, 'a.share-facebook' ) ) {
						el = event.target.parentNode;
					}

					if ( el ) {
						event.preventDefault();

						// If there's another sharing window open, close it.
						if ( typeof windowOpen !== 'undefined' ) {
							windowOpen.close();
						}
						windowOpen = window.open( el.getAttribute( 'href' ), 'wpcomfacebook', 'menubar=1,resizable=1,width=600,height=400' );
						return false;
					}
				} );
			} )();
var windowOpen;
			( function () {
				function matches( el, sel ) {
					return !! (
						el.matches && el.matches( sel ) ||
						el.msMatchesSelector && el.msMatchesSelector( sel )
					);
				}

				document.body.addEventListener( 'click', function ( event ) {
					if ( ! event.target ) {
						return;
					}

					var el;
					if ( matches( event.target, 'a.share-linkedin' ) ) {
						el = event.target;
					} else if ( event.target.parentNode && matches( event.target.parentNode, 'a.share-linkedin' ) ) {
						el = event.target.parentNode;
					}

					if ( el ) {
						event.preventDefault();

						// If there's another sharing window open, close it.
						if ( typeof windowOpen !== 'undefined' ) {
							windowOpen.close();
						}
						windowOpen = window.open( el.getAttribute( 'href' ), 'wpcomlinkedin', 'menubar=1,resizable=1,width=580,height=450' );
						return false;
					}
				} );
			} )();
//# sourceURL=sharing-js-js-after
</script>
<script id="wp-emoji-settings" type="application/json">
{"baseUrl":"https://s0.wp.com/wp-content/mu-plugins/wpcom-smileys/twemoji/2/72x72/","ext":".png","svgUrl":"https://s0.wp.com/wp-content/mu-plugins/wpcom-smileys/twemoji/2/svg/","svgExt":".svg","source":{"concatemoji":"/wp-includes/js/wp-emoji-release.min.js?m=1764078722i&ver=7.0-RC2-62280"}}
</script>
<script type="module">
/*! This file is auto-generated */
const a=JSON.parse(document.getElementById("wp-emoji-settings").textContent),o=(window._wpemojiSettings=a,"wpEmojiSettingsSupports"),s=["flag","emoji"];function i(e){try{var t={supportTests:e,timestamp:(new Date).valueOf()};sessionStorage.setItem(o,JSON.stringify(t))}catch(e){}}function c(e,t,n){e.clearRect(0,0,e.canvas.width,e.canvas.height),e.fillText(t,0,0);t=new Uint32Array(e.getImageData(0,0,e.canvas.width,e.canvas.height).data);e.clearRect(0,0,e.canvas.width,e.canvas.height),e.fillText(n,0,0);const a=new Uint32Array(e.getImageData(0,0,e.canvas.width,e.canvas.height).data);return t.every((e,t)=>e===a[t])}function p(e,t){e.clearRect(0,0,e.canvas.width,e.canvas.height),e.fillText(t,0,0);var n=e.getImageData(16,16,1,1);for(let e=0;e<n.data.length;e++)if(0!==n.data[e])return!1;return!0}function u(e,t,n,a){switch(t){case"flag":return n(e,"\ud83c\udff3\ufe0f\u200d\u26a7\ufe0f","\ud83c\udff3\ufe0f\u200b\u26a7\ufe0f")?!1:!n(e,"\ud83c\udde8\ud83c\uddf6","\ud83c\udde8\u200b\ud83c\uddf6")&&!n(e,"\ud83c\udff4\udb40\udc67\udb40\udc62\udb40\udc65\udb40\udc6e\udb40\udc67\udb40\udc7f","\ud83c\udff4\u200b\udb40\udc67\u200b\udb40\udc62\u200b\udb40\udc65\u200b\udb40\udc6e\u200b\udb40\udc67\u200b\udb40\udc7f");case"emoji":return!a(e,"\ud83e\u1fac8")}return!1}function f(e,t,n,a){let r;const o=(r="undefined"!=typeof WorkerGlobalScope&&self instanceof WorkerGlobalScope?new OffscreenCanvas(300,150):document.createElement("canvas")).getContext("2d",{willReadFrequently:!0}),s=(o.textBaseline="top",o.font="600 32px Arial",{});return e.forEach(e=>{s[e]=t(o,e,n,a)}),s}function r(e){var t=document.createElement("script");t.src=e,t.defer=!0,document.head.appendChild(t)}a.supports={everything:!0,everythingExceptFlag:!0},new Promise(t=>{let n=function(){try{var e=JSON.parse(sessionStorage.getItem(o));if("object"==typeof e&&"number"==typeof e.timestamp&&(new Date).valueOf()<e.timestamp+604800&&"object"==typeof e.supportTests)return e.supportTests}catch(e){}return null}();if(!n){if("undefined"!=typeof Worker&&"undefined"!=typeof OffscreenCanvas&&"undefined"!=typeof URL&&URL.createObjectURL&&"undefined"!=typeof Blob)try{var e="postMessage("+f.toString()+"("+[JSON.stringify(s),u.toString(),c.toString(),p.toString()].join(",")+"));",a=new Blob([e],{type:"text/javascript"});const r=new Worker(URL.createObjectURL(a),{name:"wpTestEmojiSupports"});return void(r.onmessage=e=>{i(n=e.data),r.terminate(),t(n)})}catch(e){}i(n=f(s,u,c,p))}t(n)}).then(e=>{for(const n in e)a.supports[n]=e[n],a.supports.everything=a.supports.everything&&a.supports[n],"flag"!==n&&(a.supports.everythingExceptFlag=a.supports.everythingExceptFlag&&a.supports[n]);var t;a.supports.everythingExceptFlag=a.supports.everythingExceptFlag&&!a.supports.flag,a.supports.everything||((t=a.source||{}).concatemoji?r(t.concatemoji):t.wpemoji&&t.twemoji&&(r(t.twemoji),r(t.wpemoji)))});
//# sourceURL=/wp-includes/js/wp-emoji-loader.min.js
</script>
	<iframe src='https://widgets.wp.com/likes/master.html?ver=20260510#ver=20260510' scrolling='no' id='likes-master' name='likes-master' style='display:none;'></iframe>
	<div id='likes-other-gravatars' role="dialog" aria-hidden="true" tabindex="-1"><div class="likes-text"><span>%d</span></div><ul class="wpl-avatars sd-like-gravatars"></ul></div>
	<script src="//stats.wp.com/w.js?68" defer></script> <script type="text/javascript">
_tkq = window._tkq || [];
_stq = window._stq || [];
_tkq.push(['storeContext', {'blog_id':'63261','blog_tz':'0','user_lang':'en','blog_lang':'en','user_id':'0'}]);
		// Prevent sending pageview tracking from WP-Admin pages.
		_stq.push(['view', {'blog':'63261','v':'wpcom','tz':'0','user_id':'0','post':'2103','subd':'intelligentmeasurement'}]);
		_stq.push(['extra', {'crypt':'UE40eW5QN0p8M2Y/RE1mNzc2NTVTamdsd0xoLz9RQkM2K298TXY9bERQMXc2MjhEaVZfb2wwakRoSj0mUkp1THptM1NdbkV1WjZIcU9mVWQmPUIvMlN6Jk8wW3NYVEJ3dWZOWExuWD9Ic1FKWSVBaERwSzlSdkVWcGJHeWt8cyUvclBHZFg2JjVRWGwvQi0udFFJfDZ+ZXhoZltCNDR2aEt6dForeDZfQ05qTF1wR0FpbmVBfjNiYVgyeVZvLmxxTlprVzZ8cFEsenF4eTNCUjJnXUhQNi0wbThoLkQ4a1dYYWZmbHlnRjFxVEh0LVE2fFZVc0w9PS9oflpkPWU0ZzAxajg1bzM/fj1zSzgzZjVzdFFYamlVbUs1cHFCVmJkPXBiUXA5fGI5en5CYjZXRD1wSnwmLXVpNlA0d2s='}]);
_stq.push([ 'clickTrackerInit', '63261', '2103' ]);
</script>
<noscript><img src="https://pixel.wp.com/b.gif?v=noscript" style="height:1px;width:1px;overflow:hidden;position:absolute;bottom:1px;" alt="" /></noscript>
<meta id="bilmur" property="bilmur:data" content="" data-provider="wordpress.com" data-service="simple" data-site-tz="Etc/GMT-0" data-custom-props="{&quot;logged_in&quot;:&quot;0&quot;,&quot;wptheme&quot;:&quot;pub\/blix&quot;,&quot;wptheme_is_block&quot;:&quot;0&quot;}"  >
		<script defer src="/wp-content/js/bilmur.min.js?i=17&amp;m=202619"></script> 	
<script>
(function() {
	'use strict';

	const fetches = {};
	const promises = {};
	const urls = {
		'wp-polyfill': '/wp-includes/js/dist/vendor/wp-polyfill.min.js?m=1775575011i&ver=3.15.0',
		'verbum': '/wp-content/mu-plugins/jetpack-mu-wpcom-plugin/sun/jetpack_vendor/automattic/jetpack-mu-wpcom/src/build/verbum-comments/verbum-comments.js?m=1777406400i&minify=false&ver=daa565a7d230e522cf9c'
	};
	const loaders = {
		'verbum': () => {
			fetchExternalScript('wp-polyfill');
			fetchExternalScript('verbum');
			promises['wp-polyfill'] = promises['wp-polyfill'] || loadWPScript('wp-polyfill');
			promises['verbum'] = promises['verbum'] || promises['wp-polyfill'].then( () => loadWPScript('verbum') );
			return promises['verbum'];
		},
		
	};
	const scriptExtras = {
		
	};

	window.WP_Enqueue_Dynamic_Script = {
		loadScript: (handle) => {
			if (!loaders[handle]) {
				console.error('WP_Enqueue_Dynamic_Script: unregistered script `' + handle + '`.');
			}
			return loaders[handle]();
		}
	};

	function fetchExternalScript(handle) {
		if (!urls[handle]) {
			return Promise.resolve();
		}

		fetches[handle] = fetches[handle] || fetch(urls[handle], { mode: 'no-cors' });
		return fetches[handle];
	}

	function runExtraScript(handle, type, index) {
		const id = 'wp-enqueue-dynamic-script:' + handle + ':' + type + ':' + (index + 1);
		const template = document.getElementById(id);
		if (!template) {
			return Promise.reject();
		}

		const script = document.createElement( 'script' );
		script.innerHTML = template.innerHTML;
		document.body.appendChild( script );
		return Promise.resolve();
	}

	function loadExternalScript(handle) {
		if (!urls[handle]) {
			return Promise.resolve();
		}

		return fetches[handle].then(() => {
			return new Promise((resolve, reject) => {
				const script = document.createElement('script');
				script.onload = () => resolve();
				script.onerror = (e) => reject(e);
				script.src = urls[handle];
				document.body.appendChild(script);
			});
		});
	}

	function loadExtra(handle, pos) {
		const count = (scriptExtras[handle] && scriptExtras[handle][pos]) || 0;
		let promise = Promise.resolve();

		for (let i = 0; i < count; i++) {
			promise = promise.then(() => runExtraScript(handle, pos, i));
		}

		return promise;
	}

	function loadWPScript(handle) {
		// Core loads scripts in this order. See: https://github.com/WordPress/WordPress/blob/a59eb9d39c4fcba834b70c9e8dfd64feeec10ba6/wp-includes/class-wp-scripts.php#L428.
		return loadExtra(handle, 'translations')
			.then(() => loadExtra(handle, 'before'))
			.then(() => loadExternalScript(handle))
			.then(() => loadExtra(handle, 'after'));
	}
} )();
</script>
</body>
</html>
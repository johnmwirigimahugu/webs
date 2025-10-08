/**
 * webs.js - Enterprise-Grade All-in-One Frontend Framework
 * Combines reactive JavaScript framework with integrated CSS system
 * For SPA and GUI Face Development
 * 
 * Version: 5.0.0
 * Copyright (C) 2025 by John Mwirigi Mahugu
 * MIT License
 */
(function () {
    "use strict";

    // Initialize webs namespace
    window.webs = window.webs || {};
    
    // Backward compatibility for ux- attributes
    window.ux = window.webs;

    // === CONFIGURATION ===
    webs.config = {
        baseUrl: '',
        theme: 'default',
        locale: 'en',
        pluginsDir: '/ark',
        debug: false,
        httpsOnly: true,
        csrfEnabled: true
    };

    // === INTEGRATED CSS FRAMEWORK ===
    webs.css = {
        inject: () => {
            if (document.getElementById('webs-styles')) return;
            const styleSheet = document.createElement('style');
            styleSheet.id = 'webs-styles';
            styleSheet.textContent = webs.css.getStyles();
            document.head.appendChild(styleSheet);
            webs.log('CSS Framework injected');
            webs.css.purge();
        },
        getStyles: () => `
/* === webs.JS CSS FRAMEWORK === */
:root {
    --webs-primary: #3b82f6;
    --webs-secondary: #64748b;
    --webs-success: #10b981;
    --webs-danger: #ef4444;
    --webs-warning: #f59e0b;
    --webs-info: #06b6d4;
    --webs-light: #f8fafc;
    --webs-dark: #1e293b;
    --webs-gray-200: #e5e7eb;
    --webs-font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --webs-font-size-xs: 0.75rem;
    --webs-font-size-sm: 0.875rem;
    --webs-font-size-base: 1rem;
    --webs-font-size-lg: 1.125rem;
    --webs-font-size-xl: 1.25rem;
    --webs-space-xs: 0.25rem;
    --webs-space-sm: 0.5rem;
    --webs-space-md: 1rem;
    --webs-space-lg: 1.5rem;
    --webs-radius-md: 0.375rem;
    --webs-shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --webs-transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { line-height: 1.5; -webkit-text-size-adjust: 100%; font-family: var(--webs-font-sans); }
body { margin: 0; color: var(--webs-dark); background-color: white; }
.webs-container { width: 100%; max-width: 1200px; margin: 0 auto; padding: 0 var(--webs-space-md); }
.webs-flex { display: flex; }
.webs-grid { display: grid; }
.webs-grid-cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.webs-gap-md { gap: var(--webs-space-md); }
.webs-btn { 
    display: inline-flex; 
    align-items: center; 
    padding: var(--webs-space-sm) var(--webs-space-md); 
    font-size: var(--webs-font-size-sm); 
    border-radius: var(--webs-radius-md); 
    cursor: pointer; 
    transition: var(--webs-transition);
    background-color: transparent;
    border: 1px solid transparent;
}
.webs-btn:focus { outline: 2px solid var(--webs-primary); outline-offset: 2px; }
.webs-btn[aria-disabled="true"] { opacity: 0.5; cursor: not-allowed; }
.webs-btn-primary { background-color: var(--webs-primary); color: white; border-color: var(--webs-primary); }
.webs-btn-primary:hover:not([aria-disabled="true"]) { background-color: #2563eb; }
.webs-btn-success { background-color: var(--webs-success); color: white; border-color: var(--webs-success); }
.webs-btn-danger { background-color: var(--webs-danger); color: white; border-color: var(--webs-danger); }
.webs-card { 
    background-color: white; 
    border: 1px solid var(--webs-gray-200); 
    border-radius: var(--webs-radius-md); 
    box-shadow: var(--webs-shadow-sm); 
    overflow: hidden; 
}
.webs-card-header { padding: var(--webs-space-lg); border-bottom: 1px solid var(--webs-gray-200); }
.webs-card-body { padding: var(--webs-space-lg); }
.webs-form-group { margin-bottom: var(--webs-space-md); }
.webs-label { 
    display: block; 
    font-size: var(--webs-font-size-sm); 
    font-weight: 500; 
    color: var(--webs-dark); 
}
.webs-input { 
    width: 100%; 
    padding: var(--webs-space-sm) var(--webs-space-md); 
    font-size: var(--webs-font-size-base); 
    border: 1px solid var(--webs-gray-200); 
    border-radius: var(--webs-radius-md); 
}
.webs-input:focus { outline: none; border-color: var(--webs-primary); box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1); }
.webs-alert { 
    padding: var(--webs-space-md); 
    border-radius: var(--webs-radius-md); 
    border: 1px solid; 
}
.webs-alert-success { background-color: #f0fdf4; border-color: #bbf7d0; color: #166534; }
.webs-alert-danger { background-color: #fef2f2; border-color: #fecaca; color: #991b1b; }
@media (min-width: 640px) { 
    .webs-sm\\:block { display: block; } 
    .webs-sm\\:flex { display: flex; } 
}
@media (min-width: 768px) { 
    .webs-md\\:block { display: block; } 
    .webs-md\\:grid-cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); } 
}
        `,
        purge: () => {
            const usedClasses = new Set();
            document.querySelectorAll('[class]').forEach(el => {
                el.className.split(' ').forEach(cls => usedClasses.add(cls));
            });
            const styles = webs.css.getStyles().split('\n').filter(line => {
                const match = line.match(/\.([^\s{]+)/);
                return !match || usedClasses.has(match[1]);
            }).join('\n');
            const styleSheet = document.getElementById('webs-styles');
            if (styleSheet) styleSheet.textContent = styles;
            webs.log('CSS purged');
        }
    };

    // === DOM UTILITIES ===
    webs.dom = {
        get: sel => document.querySelector(sel),
        getAll: sel => document.querySelectorAll(sel),
        show: sel => webs.dom.getAll(sel).forEach(el => el && (el.style.display = 'block')),
        hide: sel => webs.dom.getAll(sel).forEach(el => el && (el.style.display = 'none')),
        addClass: (sel, cls) => webs.dom.getAll(sel).forEach(el => el?.classList.add(cls)),
        removeClass: (sel, cls) => webs.dom.getAll(sel).forEach(el => el?.classList.remove(cls)),
        toggleClass: (sel, cls) => webs.dom.getAll(sel).forEach(el => el?.classList.toggle(cls)),
        on: (sel, event, handler, opts = {}) => {
            const delegateHandler = e => {
                const target = e.target.closest(sel);
                if (target) handler.call(target, e);
            };
            document.addEventListener(event, delegateHandler, opts);
            return () => document.removeEventListener(event, delegateHandler);
        },
        batchUpdate: (updates) => {
            requestAnimationFrame(() => {
                updates.forEach(({ sel, prop, val }) => {
                    webs.dom.getAll(sel).forEach(el => el && (el[prop] = val));
                });
            });
        },
        template: (tpl, data) => {
            let html = typeof tpl === 'string' ? tpl : tpl.cloneNode(true).innerHTML;
            for (const key in data) {
                html = html.replace(new RegExp(`{{${key}}}`, 'g'), webs.html.escape(data[key]));
            }
            return html;
        },
        includeHtml: async (url, target, cb) => {
            const el = typeof target === 'string' ? webs.dom.get(target) : target;
            if (!el) return webs.log('includeHtml: Target not found:', target);
            el.innerHTML = '<div class="webs-spinner webs-m-auto"></div>';
            try {
                const res = await fetch(url);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                el.innerHTML = await res.text();
                webs.initElement(el);
                cb?.();
            } catch (e) {
                webs.log('includeHtml Error:', url, e);
                el.innerHTML = `<div class="webs-alert webs-alert-danger">${webs.i18n.t('error.html_include')}</div>`;
            }
        }
    };

    // === VIRTUAL DOM ===
    webs.vdom = {
        _cache: new WeakMap(),
        _memo: new Map(),
        diff: (oldNode, newNode, parent) => {
            if (!oldNode || !newNode) return parent.replaceChild(newNode, oldNode || parent.firstChild);
            const memoKey = `${oldNode?.outerHTML || ''}:${newNode?.outerHTML || ''}`;
            if (webs.vdom._memo.has(memoKey)) return;
            if (oldNode.nodeType === 3 && newNode.nodeType === 3) {
                if (oldNode.textContent !== newNode.textContent) oldNode.textContent = newNode.textContent;
                webs.vdom._memo.set(memoKey, true);
                return;
            }
            if (oldNode.tagName !== newNode.tagName) {
                parent.replaceChild(newNode, oldNode);
                webs.vdom._memo.set(memoKey, true);
                return;
            }
            const oldAttrs = oldNode.attributes, newAttrs = newNode.attributes;
            for (const attr of newAttrs) oldNode.setAttribute(attr.name, attr.value);
            for (const attr of oldAttrs) if (!newNode.hasAttribute(attr.name)) oldNode.removeAttribute(attr.name);
            const oldKids = Array.from(oldNode.childNodes), newKids = Array.from(newNode.childNodes);
            for (let i = 0; i < Math.max(oldKids.length, newKids.length); i++) {
                webs.vdom.diff(oldKids[i], newKids[i], oldNode);
            }
            webs.vdom._memo.set(memoKey, true);
        },
        render: (el, data) => {
            const key = el.getAttribute('webs-key') || el.getAttribute('ux-key') || el.id;
            const html = webs.dom.template(el, data);
            const newNode = document.createElement('div');
            newNode.innerHTML = html;
            const newEl = newNode.firstChild;
            if (key && webs.vdom._cache.has(key)) {
                webs.vdom.diff(webs.vdom._cache.get(key), newEl, el.parentNode);
            } else {
                el.parentNode.replaceChild(newEl, el);
            }
            webs.vdom._cache.set(key, newEl);
            webs.initElement(newEl);
        }
    };

    // === AJAX AND HTTP ===
    webs.ajax = {
        middleware: [],
        addMiddleware: (fn) => webs.ajax.middleware.push(fn),
        request: async (method, url, data, success, error, opts = {}) => {
            webs.perf.start('ajax');
            try {
                if (webs.config.httpsOnly && !url.startsWith('https://') && !url.startsWith('/')) {
                    throw new Error(webs.i18n.t('error.https_required'));
                }
                const headers = { 
                    'X-Requested-With': 'XMLHttpRequest', 
                    ...(webs.config.csrfEnabled ? { 'X-CSRF-Token': webs.auth.getCsrfToken() } : {}), 
                    ...opts.headers 
                };
                if (!headers['Content-Type'] && data && ['POST', 'PUT', 'PATCH'].includes(method)) {
                    headers['Content-Type'] = 'application/json';
                }
                let body = data;
                if (data && headers['Content-Type'].includes('json')) {
                    body = JSON.stringify(data);
                } else if (data && typeof data === 'object' && !(data instanceof FormData)) {
                    body = Object.entries(data).map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join('&');
                }
                const request = { method, url, headers, body };
                for (const mw of webs.ajax.middleware) {
                    await mw(request);
                }
                const response = await fetch(url, { method, headers, body });
                const result = response.headers.get('content-type')?.includes('json') ? await response.json() : await response.text();
                if (response.ok) {
                    success?.(result);
                    webs.events.emit(`ajax:${response.status}`, result);
                } else {
                    throw new Error(`HTTP ${response.status}`);
                }
            } catch (e) {
                error?.(e);
                webs.log('AJAX Error:', e);
                webs.flash.add('danger', e.message);
                webs.events.emit('ajax:error', e);
            }
            webs.perf.end('ajax');
        },
        jsonp: (url, paddingName = 'callback', cb) => {
            const script = document.createElement('script');
            const padding = `websjsonp_${Date.now()}${Math.floor(Math.random() * 10000)}`;
            window[padding] = data => {
                cb?.(data);
                document.head.removeChild(script);
                delete window[padding];
            };
            script.src = url + (url.includes('?') ? '&' : '?') + `${paddingName}=${padding}`;
            script.onerror = () => webs.events.emit('ajax:error', new Error(webs.i18n.t('error.jsonp')));
            document.head.appendChild(script);
        }
    };

    // === FORM SERIALIZATION ===
    webs.form = {
        serialize: form => {
            const formData = new FormData(form);
            const json = {};
            formData.forEach((val, key) => {
                json[key] = json[key] ? (Array.isArray(json[key]) ? [...json[key], val] : [json[key], val]) : val;
            });
            return json;
        }
    };

    // === WEBSOCKET AND SSE ===
    webs.ws = {
        sockets: new Map(),
        connect: (url, target, swap = 'innerHTML') => {
            if (webs.ws.sockets.has(url)) return;
            const socket = new WebSocket(url);
            webs.ws.sockets.set(url, socket);
            const el = typeof target === 'string' ? webs.dom.get(target) : target;
            socket.onmessage = e => {
                webs.dom.batchUpdate([{ sel: target, prop: 'innerHTML', val: e.data }]);
                webs.initElement(el);
                webs.events.emit('ws-message', { url, data: e.data });
            };
            socket.onerror = e => webs.log('WebSocket Error:', url, e);
            socket.onclose = () => webs.ws.sockets.delete(url);
        }
    };
    webs.sse = {
        sources: new Map(),
        connect: (url, target, swap = 'innerHTML') => {
            if (webs.sse.sources.has(url)) return;
            const source = new EventSource(url);
            webs.sse.sources.set(url, source);
            const el = typeof target === 'string' ? webs.dom.get(target) : target;
            source.onmessage = e => {
                webs.dom.batchUpdate([{ sel: target, prop: 'innerHTML', val: e.data }]);
                webs.initElement(el);
                webs.events.emit('sse-message', { url, data: e.data });
            };
            source.onerror = () => {
                webs.log('SSE Error:', url);
                webs.sse.sources.delete(url);
            };
        }
    };

    // === STATE MANAGEMENT ===
    webs.store = {
        _data: new Proxy({}, {
            set(target, key, val) {
                const old = target[key];
                target[key] = val;
                webs.events.emit('store', { key, val, old });
                return true;
            }
        }),
        set: (key, val) => webs.store._data[key] = val,
        get: key => webs.store._data[key]
    };

    // === AUTHENTICATION ===
    webs.auth = {
        _token: null,
        login: async (url, credentials) => {
            try {
                const response = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(credentials)
                });
                if (response.ok) {
                    webs.auth._token = (await response.json()).token;
                    sessionStorage.setItem('webs_token', webs.auth._token);
                    webs.events.emit('auth:login', webs.auth._token);
                } else {
                    throw new Error(webs.i18n.t('error.auth_failed'));
                }
            } catch (e) {
                webs.flash.add('danger', e.message);
            }
        },
        getCsrfToken: () => {
            if (!webs.auth._token) webs.auth._token = sessionStorage.getItem('webs_token');
            return webs.auth._token || btoa(Math.random().toString());
        }
    };

    // === INTERNATIONALIZATION ===
    webs.i18n = {
        _translations: new Map(),
        load: async (locale) => {
            try {
                const response = await fetch(`/i18n/${locale}.json`);
                if (!response.ok) throw new Error('Failed to load translations');
                webs.i18n._translations.set(locale, await response.json());
                webs.config.locale = locale;
                webs.events.emit('i18n:loaded', locale);
                webs.dom.getAll('[webs-i18n], [ux-i18n]').forEach(el => {
                    const key = el.getAttribute('webs-i18n') || el.getAttribute('ux-i18n');
                    el.textContent = webs.i18n.t(key);
                });
            } catch (e) {
                webs.log('i18n load error:', e);
                webs.flash.add('danger', webs.i18n.t('error.i18n_load'));
            }
        },
        t: (key) => {
            const translations = webs.i18n._translations.get(webs.config.locale) || {};
            return translations[key] || key;
        }
    };

    // === MICRO-LIBRARY AUTOLOADER ===
    webs.ark = {
        _loaded: new Map(),
        load: async (basePath = webs.config.pluginsDir) => {
            webs.perf.start('ark');
            try {
                const response = await fetch(`${basePath}/ark.json`);
                if (!response.ok) throw new Error('Failed to load ark.json');
                const config = await response.json();
                for (const lib of config.libraries) {
                    if (!webs.ark._loaded.has(lib.name)) {
                        const script = document.createElement('script');
                        script.src = `${basePath}/${lib.file}`;
                        script.async = true;
                        document.head.appendChild(script);
                        await new Promise(resolve => {
                            script.onload = () => {
                                if (window[lib.namespace]) {
                                    webs.plugins.register(lib.name, window[lib.namespace]);
                                    webs.ark._loaded.set(lib.name, window[lib.namespace]);
                                    webs.events.emit('ark:loaded', { name: lib.name, module: window[lib.namespace] });
                                    if (lib.init) window[lib.namespace][lib.init]();
                                }
                                resolve();
                            };
                            script.onerror = () => {
                                webs.log(`Failed to load library: ${lib.name}`);
                                resolve();
                            };
                        });
                    }
                }
            } catch (e) {
                webs.log('Ark load error:', e);
                webs.flash.add('danger', webs.i18n.t('error.ark_load'));
            }
            webs.perf.end('ark');
        },
        get: name => webs.ark._loaded.get(name)
    };

    // === COMPONENT SYSTEM ===
    webs.components = {
        _hooks: new Map(),
        _components: new Map(),
        register: (name, config) => {
            webs.components._components.set(name, config);
            if (config.template) {
                webs.store.set(`__component__${name}`, config.template);
            }
            ['onMount', 'onUpdate', 'onUnmount'].forEach(hook => {
                if (config[hook]) webs.components._hooks.set(`${name}:${hook}`, config[hook]);
            });
        },
        load: async (url) => {
            webs.perf.start('component_load');
            try {
                const response = await fetch(url);
                const manifest = await response.json();
                const baseUrl = url.substring(0, url.lastIndexOf('/'));
                for (const script of manifest.scripts || []) {
                    const scriptEl = document.createElement('script');
                    scriptEl.src = `${baseUrl}/${script}`;
                    document.head.appendChild(scriptEl);
                    await new Promise(resolve => scriptEl.onload = resolve);
                }
                if (manifest.styles) {
                    const styleEl = document.createElement('link');
                    styleEl.rel = 'stylesheet';
                    styleEl.href = `${baseUrl}/${manifest.styles}`;
                    document.head.appendChild(styleEl);
                }
                if (manifest.template) {
                    const template = await (await fetch(`${baseUrl}/${manifest.template}`)).text();
                    webs.components.register(manifest.name, { ...manifest, template });
                    webs.events.emit('component:loaded', manifest.name);
                }
            } catch (e) {
                webs.log('Component load error:', e);
                webs.flash.add('danger', webs.i18n.t('error.component_load'));
            }
            webs.perf.end('component_load');
        },
        triggerHook: (name, hook, el) => {
            const handler = webs.components._hooks.get(`${name}:${hook}`);
            if (handler) handler(el);
        }
    };

    // === PLUGINS ===
    webs.plugins = {
        _directives: new Map(),
        register: (name, handler) => webs.plugins._directives.set(name, handler),
        apply: (el, directive, value) => {
            const handler = webs.plugins._directives.get(directive);
            if (handler) handler(el, value);
        }
    };

    // Built-in Tooltip Plugin
    webs.plugins.register('tooltip', (el, value) => {
        const tooltip = document.createElement('span');
        tooltip.className = 'webs-badge webs-absolute webs-hidden webs-bg-dark webs-text-white webs-p-sm webs-rounded';
        tooltip.style.zIndex = '10';
        tooltip.textContent = webs.i18n.t(value);
        el.appendChild(tooltip);
        el.setAttribute('aria-describedby', 'tooltip');
        webs.dom.on(el, 'mouseenter', () => webs.dom.show(tooltip));
        webs.dom.on(el, 'mouseleave', () => webs.dom.hide(tooltip));
        webs.dom.on(el, 'focus', () => webs.dom.show(tooltip));
        webs.dom.on(el, 'blur', () => webs.dom.hide(tooltip));
    });

    // === CHART MODULE ===
    webs.chart = {
        types: ['line', 'bar', 'pie'],
        render: (selector, type, config) => {
            const el = webs.dom.get(selector);
            if (!el || !webs.chart.types.includes(type)) {
                webs.log(`Chart error: Invalid selector (${selector}) or type (${type})`);
                return;
            }
            const defaults = {
                title: webs.i18n.t('chart.title'),
                data: [],
                xLabel: '',
                yLabel: '',
                width: 300,
                height: 200
            };
            const cfg = { ...defaults, ...config };
            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.setAttribute('width', cfg.width);
            svg.setAttribute('height', cfg.height);
            svg.setAttribute('class', 'webs-card webs-p-md');
            svg.setAttribute('aria-label', cfg.title);
            el.innerHTML = '';
            el.appendChild(svg);
            // Simplified chart rendering (same as sumo.js)
            // Add WebAssembly placeholder for future optimization
            webs.events.emit('chart:render', { selector, type, config });
        }
    };

    // === EVENT SYSTEM ===
    webs.events = {
        _listeners: new Map(),
        on: (evt, cb) => {
            const listeners = webs.events._listeners.get(evt) || [];
            webs.events._listeners.set(evt, listeners.concat(cb));
        },
        emit: (evt, data) => {
            webs.events._listeners.get(evt)?.forEach(cb => cb(data));
        },
        off: (evt, cb) => {
            webs.events._listeners.set(evt, webs.events._listeners.get(evt)?.filter(c => c !== cb));
        }
    };

    // === FLASH MESSAGES ===
    webs.flash = {
        add: (type, message, duration = 3000) => {
            const alert = document.createElement('div');
            alert.className = `webs-alert webs-alert-${type} webs-fixed webs-top-0 webs-right-0 webs-m-md webs-z-50`;
            alert.setAttribute('role', 'alert');
            alert.textContent = webs.i18n.t(message);
            document.body.appendChild(alert);
            setTimeout(() => {
                alert.remove();
            }, duration);
            return alert;
        }
    };

    // === PERFORMANCE MONITORING ===
    webs.perf = {
        _timers: new Map(),
        start: key => webs.perf._timers.set(key, performance.now()),
        end: key => {
            const start = webs.perf._timers.get(key);
            if (start) webs.log(`Perf: ${key} took ${performance.now() - start}ms`);
        }
    };

    // === TESTING ===
    webs.test = {
        assert: (cond, msg) => { if (!cond) throw new Error(`Test failed: ${msg}`); },
        run: async tests => {
            let passed = 0, failed = 0;
            for (const [name, test] of Object.entries(tests)) {
                try {
                    await test();
                    webs.log(`Test passed: ${name}`);
                    passed++;
                } catch (e) {
                    webs.log(`Test failed: ${name}`, e);
                    failed++;
                }
            }
            webs.log(`Tests: ${passed} passed, ${failed} failed`);
            return { passed, failed };
        }
    };

    // === INITIALIZATION ===
    webs.init = async () => {
        webs.perf.start('init');
        webs.css.inject();
        await webs.i18n.load(webs.config.locale);
        await webs.ark.load();
        webs.dom.getAll('[webs-data], [ux-data], [webs-on\\:], [ux-on\\:], [webs-get], [ux-get], [webs-post], [ux-post], [webs-put], [ux-put], [webs-delete], [ux-delete], [webs-component], [ux-component], [webs-i18n], [ux-i18n]').forEach(webs.initElement);
        webs.log('webs.js v5.0.0 initialized');
        webs.perf.end('init');
    };

    webs.initElement = el => {
        webs.perf.start('initElement');
        try {
            const dataAttr = el.getAttribute('webs-data') || el.getAttribute('ux-data');
            if (dataAttr) {
                const data = JSON.parse(dataAttr);
                el.webs = new Proxy(data, {
                    set(t, p, v) {
                        t[p] = v;
                        const updates = [
                            { sel: `[webs-bind="${p}"], [ux-bind="${p}"]`, prop: 'value', val: v },
                            { sel: `[webs-text="${p}"], [ux-text="${p}"]`, prop: 'innerText', val: v },
                            { sel: `[webs-show="${p}"], [ux-show="${p}"]`, prop: 'style.display', val: v ? 'block' : 'none' }
                        ];
                        webs.dom.batchUpdate(updates);
                        if (el.getAttribute('webs-persist') || el.getAttribute('ux-persist')) {
                            sessionStorage.setItem(el.getAttribute('webs-persist') || el.getAttribute('ux-persist'), JSON.stringify(t));
                        }
                        webs.vdom.render(el, t);
                        webs.components.triggerHook(el.getAttribute('webs-component') || el.getAttribute('ux-component'), 'onUpdate', el);
                        return true;
                    }
                });
                webs.dom.getAll('[webs-bind], [ux-bind]', el).forEach(e => {
                    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.tagName)) {
                        webs.dom.on(e, 'input', () => e.webs[e.getAttribute('webs-bind') || e.getAttribute('ux-bind')] = e.value);
                    }
                });
                webs.vdom.render(el, el.webs);
            }
            ['get', 'post', 'put', 'delete'].forEach(method => {
                webs.dom.getAll(`[webs-${method}], [ux-${method}]`, el).forEach(e => {
                    webs.dom.on(e, e.tagName === 'FORM' ? 'submit' : 'click', async ev => {
                        ev.preventDefault();
                        const url = e.getAttribute(`webs-${method}`) || e.getAttribute(`ux-${method}`);
                        const target = e.getAttribute('webs-swap') || e.getAttribute('ux-swap') || '#content';
                        const swap = e.getAttribute('webs-swap-strategy') || e.getAttribute('ux-swap-strategy') || 'innerHTML';
                        const data = e.tagName === 'FORM' ? webs.form.serialize(e) : JSON.parse(e.getAttribute('webs-data-payload') || e.getAttribute('ux-data-payload') || '{}');
                        await webs.ajax.request(method.toUpperCase(), url, data, 
                            r => webs.dom.batchUpdate([{ sel: target, prop: 'innerHTML', val: r }]),
                            e => webs.flash.add('danger', e.message)
                        );
                    });
                });
            });
            webs.dom.getAll('[webs-custom\\:], [ux-custom\\:]', el).forEach(e => {
                e.getAttributeNames().filter(n => n.startsWith('webs-custom:') || n.startsWith('ux-custom:')).forEach(n => {
                    const dir = n.replace(/^(webs|ux)-custom:/, '');
                    webs.plugins.apply(e, dir, e.getAttribute(n));
                });
            });
            if (el.hasAttribute('webs-component') || el.hasAttribute('ux-component')) {
                webs.components.triggerHook(el.getAttribute('webs-component') || el.getAttribute('ux-component'), 'onMount', el);
            }
        } catch (e) {
            webs.log('Error in initElement:', e);
            webs.flash.add('danger', webs.i18n.t('error.init_element'));
        }
        webs.perf.end('initElement');
    };

    // === HTML UTILITIES ===
    webs.html = {
        escape: str => {
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }
    };

    // === LOGGING ===
    webs.log = (...args) => {
        if (webs.config.debug) console.log('[webs.js]', ...args);
    };

    // === TYPESCRIPT DEFINITIONS ===
    webs.types = `
        declare namespace webs {
            interface Config {
                baseUrl: string;
                theme: string;
                locale: string;
                pluginsDir: string;
                debug: boolean;
                httpsOnly: boolean;
                csrfEnabled: boolean;
            }
            interface Dom {
                get: (sel: string) => HTMLElement | null;
                getAll: (sel: string) => NodeList;
                show: (sel: string | HTMLElement) => void;
                hide: (sel: string | HTMLElement) => void;
                batchUpdate: (updates: { sel: string, prop: string, val: any }[]) => void;
                template: (tpl: string | HTMLElement, data: any) => string;
            }
            interface VDom {
                render: (el: HTMLElement, data: any) => void;
            }
            interface Ajax {
                request: (method: string, url: string, data: any, success?: (result: any) => void, error?: (e: Error) => void, opts?: any) => Promise<void>;
                jsonp: (url: string, paddingName?: string, cb?: (data: any) => void) => void;
            }
            interface Auth {
                login: (url: string, credentials: any) => Promise<void>;
                getCsrfToken: () => string;
            }
            interface I18n {
                load: (locale: string) => Promise<void>;
                t: (key: string) => string;
            }
            interface Ark {
                load: (basePath?: string) => Promise<void>;
                get: (name: string) => any;
            }
            interface Components {
                register: (name: string, config: { template?: string, onMount?: (el: HTMLElement) => void, onUpdate?: (el: HTMLElement) => void, onUnmount?: (el: HTMLElement) => void }) => void;
                load: (url: string) => Promise<void>;
            }
            interface Plugins {
                register: (name: string, handler: (el: HTMLElement, value: any) => void) => void;
                apply: (el: HTMLElement, directive: string, value: any) => void;
            }
        }
    `;

    // === CLI TOOL (Placeholder for Documentation) ===
    webs.cli = {
        init: () => webs.log('webs-cli: Run `npx webs-cli init` to scaffold a new project'),
        generate: (type, name) => webs.log(`webs-cli: Run 'npx webs-cli generate ${type} ${name}' to create a new ${type}`)
    };

    // === INITIALIZATION ===
    document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', webs.init) : webs.init();

    /* johnmahugu at protonmail dot com */
    /* repository and documentation: webs.js.org */
    /* Thanks יהוה by Kesh EOF (2468) (July 19, 2025 3:56 AM EAT) (UNIX TIME STAMP: 1755485760) (UUID: ad0dcfc3-c59b-4f2c-895e-5672379834a7) */
})();
/*!
 * Webs.js v1.0 - Enterprise-Grade Frontend Framework
 * Enhanced from go.js with Vue.js-like capabilities
 * Prefix: cx-
 * Single file, world-class solution
 */

(function() {
    'use strict';

    // ===== CONFIGURATION =====
    const config = {
        debug: false,
        prefix: 'cx-',
        version: '1.0.0',
        plugins: [],
        components: {},
        directives: {},
        transitions: {},
        filters: {}
    };

    // ===== UTILITY FUNCTIONS =====
    const utils = {
        // Generate unique ID
        uuid() {
            return 'cx-' + Math.random().toString(36).substr(2, 9);
        },

        // Deep merge objects
        merge(target, source) {
            const result = { ...target };
            for (const key in source) {
                if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                    result[key] = this.merge(result[key] || {}, source[key]);
                } else {
                    result[key] = source[key];
                }
            }
            return result;
        },

        // Debounce function
        debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        },

        // Throttle function
        throttle(func, limit) {
            let inThrottle;
            return function() {
                const args = arguments;
                const context = this;
                if (!inThrottle) {
                    func.apply(context, args);
                    inThrottle = true;
                    setTimeout(() => inThrottle = false, limit);
                }
            };
        },

        // Check if element is in viewport
        isInViewport(element) {
            const rect = element.getBoundingClientRect();
            return (
                rect.top >= 0 &&
                rect.left >= 0 &&
                rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
                rect.right <= (window.innerWidth || document.documentElement.clientWidth)
            );
        },

        // Escape HTML
        escapeHtml(text) {
            const map = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#039;'
            };
            return text.replace(/[&<>"']/g, m => map[m]);
        },

        // Format date
        formatDate(date, format = 'YYYY-MM-DD') {
            const d = new Date(date);
            const year = d.getFullYear();
            const month = String(d.getMonth() + 1).padStart(2, '0');
            const day = String(d.getDate()).padStart(2, '0');
            const hours = String(d.getHours()).padStart(2, '0');
            const minutes = String(d.getMinutes()).padStart(2, '0');
            const seconds = String(d.getSeconds()).padStart(2, '0');

            return format
                .replace('YYYY', year)
                .replace('MM', month)
                .replace('DD', day)
                .replace('HH', hours)
                .replace('mm', minutes)
                .replace('ss', seconds);
        },

        // Get query parameter
        getQueryParam(name) {
            const urlParams = new URLSearchParams(window.location.search);
            return urlParams.get(name);
        },

        // Set cookie
        setCookie(name, value, days = 7) {
            const expires = new Date();
            expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));
            document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/`;
        },

        // Get cookie
        getCookie(name) {
            const nameEQ = name + "=";
            const ca = document.cookie.split(';');
            for (let i = 0; i < ca.length; i++) {
                let c = ca[i];
                while (c.charAt(0) === ' ') c = c.substring(1, c.length);
                if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
            }
            return null;
        },

        // Delete cookie
        deleteCookie(name) {
            document.cookie = `${name}=;expires=Thu, 01 Jan 1970 00:00:00 UTC;path=/;`;
        },

        // Make AJAX request
        ajax(options) {
            return new Promise((resolve, reject) => {
                const defaults = {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Requested-With': 'XMLHttpRequest'
                    },
                    credentials: 'same-origin'
                };

                const settings = utils.merge(defaults, options);

                fetch(settings.url, settings)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP error! Status: ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(data => resolve(data))
                    .catch(error => reject(error));
            });
        },

        // Find elements by selector
        find(selector, context = document) {
            return context.querySelectorAll(selector);
        },

        // Find single element by selector
        findOne(selector, context = document) {
            return context.querySelector(selector);
        },

        // Add event listener
        on(element, event, handler, options = {}) {
            element.addEventListener(event, handler, options);
        },

        // Remove event listener
        off(element, event, handler) {
            element.removeEventListener(event, handler);
        },

        // Add class
        addClass(element, className) {
            element.classList.add(className);
        },

        // Remove class
        removeClass(element, className) {
            element.classList.remove(className);
        },

        // Toggle class
        toggleClass(element, className) {
            element.classList.toggle(className);
        },

        // Has class
        hasClass(element, className) {
            return element.classList.contains(className);
        },

        // Set attribute
        setAttr(element, name, value) {
            element.setAttribute(name, value);
        },

        // Get attribute
        getAttr(element, name) {
            return element.getAttribute(name);
        },

        // Remove attribute
        removeAttr(element, name) {
            element.removeAttribute(name);
        },

        // Set style
        setStyle(element, property, value) {
            element.style[property] = value;
        },

        // Get style
        getStyle(element, property) {
            return window.getComputedStyle(element)[property];
        },

        // Create element
        create(tag, attributes = {}, children = []) {
            const element = document.createElement(tag);
            
            for (const [key, value] of Object.entries(attributes)) {
                if (key === 'className') {
                    element.className = value;
                } else if (key === 'innerHTML') {
                    element.innerHTML = value;
                } else if (key === 'textContent') {
                    element.textContent = value;
                } else {
                    element.setAttribute(key, value);
                }
            }
            
            children.forEach(child => {
                if (typeof child === 'string') {
                    element.appendChild(document.createTextNode(child));
                } else {
                    element.appendChild(child);
                }
            });
            
            return element;
        },

        // Empty element
        empty(element) {
            while (element.firstChild) {
                element.removeChild(element.firstChild);
            }
        },

        // Remove element
        remove(element) {
            if (element.parentNode) {
                element.parentNode.removeChild(element);
            }
        },

        // Clone element
        clone(element, deep = true) {
            return element.cloneNode(deep);
        },

        // Get parent element
        parent(element, selector) {
            const parent = element.parentElement;
            if (!selector || parent.matches(selector)) {
                return parent;
            }
            return this.parent(parent, selector);
        },

        // Get next sibling
        next(element, selector) {
            let next = element.nextElementSibling;
            while (next) {
                if (!selector || next.matches(selector)) {
                    return next;
                }
                next = next.nextElementSibling;
            }
            return null;
        },

        // Get previous sibling
        prev(element, selector) {
            let prev = element.previousElementSibling;
            while (prev) {
                if (!selector || prev.matches(selector)) {
                    return prev;
                }
                prev = prev.previousElementSibling;
            }
            return null;
        },

        // Get children
        children(element, selector) {
            const children = Array.from(element.children);
            return selector ? children.filter(child => child.matches(selector)) : children;
        },

        // Get index of element in parent
        index(element) {
            return Array.from(element.parentNode.children).indexOf(element);
        },

        // Animate element
        animate(element, keyframes, options = {}) {
            return element.animate(keyframes, options);
        },

        // Scroll to element
        scrollTo(element, options = {}) {
            const defaults = {
                behavior: 'smooth',
                block: 'start'
            };
            
            element.scrollIntoView(utils.merge(defaults, options));
        },

        // Scroll to top
        scrollToTop(options = {}) {
            const defaults = {
                top: 0,
                behavior: 'smooth'
            };
            
            window.scrollTo(utils.merge(defaults, options));
        },

        // Get scroll position
        getScrollPosition() {
            return {
                x: window.pageXOffset || document.documentElement.scrollLeft,
                y: window.pageYOffset || document.documentElement.scrollTop
            };
        },

        // Set scroll position
        setScrollPosition(x, y) {
            window.scrollTo(x, y);
        },

        // Get element position
        getPosition(element) {
            const rect = element.getBoundingClientRect();
            return {
                top: rect.top + window.pageYOffset,
                left: rect.left + window.pageXOffset,
                width: rect.width,
                height: rect.height
            };
        },

        // Get element dimensions
        getDimensions(element) {
            return {
                width: element.offsetWidth,
                height: element.offsetHeight,
                innerWidth: element.clientWidth,
                innerHeight: element.clientHeight
            };
        },

        // Check if element is visible
        isVisible(element) {
            return !!(element.offsetWidth || element.offsetHeight || element.getClientRects().length);
        },

        // Show element
        show(element) {
            element.style.display = '';
        },

        // Hide element
        hide(element) {
            element.style.display = 'none';
        },

        // Toggle element visibility
        toggle(element) {
            if (utils.isVisible(element)) {
                utils.hide(element);
            } else {
                utils.show(element);
            }
        },

        // Fade in element
        fadeIn(element, duration = 300) {
            element.style.opacity = 0;
            element.style.display = '';
            
            (function fade() {
                let val = parseFloat(element.style.opacity);
                if (!((val += 0.1) > 1)) {
                    element.style.opacity = val;
                    requestAnimationFrame(fade);
                }
            })();
        },

        // Fade out element
        fadeOut(element, duration = 300) {
            (function fade() {
                if ((element.style.opacity -= 0.1) < 0) {
                    element.style.display = 'none';
                } else {
                    requestAnimationFrame(fade);
                }
            })();
        },

        // Slide up element
        slideUp(element, duration = 300) {
            element.style.height = element.offsetHeight + 'px';
            element.style.transitionProperty = 'height, margin, padding';
            element.style.transitionDuration = duration + 'ms';
            element.offsetHeight; // Trigger reflow
            element.style.overflow = 'hidden';
            element.style.height = 0;
            element.style.paddingTop = 0;
            element.style.paddingBottom = 0;
            element.style.marginTop = 0;
            element.style.marginBottom = 0;
            
            setTimeout(() => {
                element.style.display = 'none';
                element.style.removeProperty('height');
                element.style.removeProperty('padding-top');
                element.style.removeProperty('padding-bottom');
                element.style.removeProperty('margin-top');
                element.style.removeProperty('margin-bottom');
                element.style.removeProperty('overflow');
                element.style.removeProperty('transition-duration');
                element.style.removeProperty('transition-property');
            }, duration);
        },

        // Slide down element
        slideDown(element, duration = 300) {
            element.style.removeProperty('display');
            let display = window.getComputedStyle(element).display;
            
            if (display === 'none') {
                display = 'block';
            }
            
            element.style.display = display;
            const height = element.offsetHeight;
            element.style.overflow = 'hidden';
            element.style.height = 0;
            element.style.paddingTop = 0;
            element.style.paddingBottom = 0;
            element.style.marginTop = 0;
            element.style.marginBottom = 0;
            element.offsetHeight; // Trigger reflow
            element.style.transitionProperty = 'height, margin, padding';
            element.style.transitionDuration = duration + 'ms';
            element.style.height = height + 'px';
            element.style.removeProperty('padding-top');
            element.style.removeProperty('padding-bottom');
            element.style.removeProperty('margin-top');
            element.style.removeProperty('margin-bottom');
            
            setTimeout(() => {
                element.style.removeProperty('height');
                element.style.removeProperty('overflow');
                element.style.removeProperty('transition-duration');
                element.style.removeProperty('transition-property');
            }, duration);
        },

        // Slide toggle element
        slideToggle(element, duration = 300) {
            if (window.getComputedStyle(element).display === 'none') {
                utils.slideDown(element, duration);
            } else {
                utils.slideUp(element, duration);
            }
        }
    };

    // ===== REACTIVE SYSTEM =====
    class Reactive {
        constructor(data = {}) {
            this.data = data;
            this.watchers = {};
            this.makeReactive(data);
        }

        makeReactive(obj, path = '') {
            const self = this;
            
            return new Proxy(obj, {
                get(target, key) {
                    const value = target[key];
                    
                    if (typeof value === 'object' && value !== null) {
                        return self.makeReactive(value, path ? `${path}.${key}` : key);
                    }
                    
                    return value;
                },
                
                set(target, key, value) {
                    const oldValue = target[key];
                    target[key] = value;
                    
                    const fullPath = path ? `${path}.${key}` : key;
                    self.notify(fullPath, value, oldValue);
                    
                    return true;
                }
            });
        }

        watch(key, callback) {
            if (!this.watchers[key]) {
                this.watchers[key] = [];
            }
            this.watchers[key].push(callback);
        }

        notify(key, newValue, oldValue) {
            if (this.watchers[key]) {
                this.watchers[key].forEach(callback => callback(newValue, oldValue));
            }
        }

        set(key, value) {
            const keys = key.split('.');
            let current = this.data;
            
            for (let i = 0; i < keys.length - 1; i++) {
                if (!current[keys[i]]) {
                    current[keys[i]] = {};
                }
                current = current[keys[i]];
            }
            
            current[keys[keys.length - 1]] = value;
        }

        get(key) {
            const keys = key.split('.');
            let current = this.data;
            
            for (const k of keys) {
                if (!current[k]) {
                    return undefined;
                }
                current = current[k];
            }
            
            return current;
        }
    }

    // ===== COMPONENT SYSTEM =====
    class Component {
        constructor(options = {}) {
            this.name = options.name || 'component';
            this.template = options.template || '';
            this.data = options.data || (() => ({}));
            this.methods = options.methods || {};
            this.computed = options.computed || {};
            this.watch = options.watch || {};
            this.mounted = options.mounted || null;
            this.beforeDestroy = options.beforeDestroy || null;
            this.props = options.props || [];
            this.el = options.el || null;
            this.parent = options.parent || null;
            
            this.state = new Reactive();
            this.isMounted = false;
            this.children = [];
            this.eventListeners = [];
        }

        init() {
            // Initialize data
            if (typeof this.data === 'function') {
                this.state.data = this.state.makeReactive(this.data());
            } else {
                this.state.data = this.state.makeReactive(this.data);
            }

            // Add computed properties
            for (const [key, fn] of Object.entries(this.computed)) {
                Object.defineProperty(this.state.data, key, {
                    get: fn.bind(this),
                    enumerable: true,
                    configurable: true
                });
            }

            // Add watchers
            for (const [key, handler] of Object.entries(this.watch)) {
                this.state.watch(key, handler.bind(this));
            }

            // Bind methods
            for (const [key, method] of Object.entries(this.methods)) {
                this.methods[key] = method.bind(this);
            }

            // Set up event listeners
            this.setupEventListeners();

            // Render template
            if (this.el && this.template) {
                this.render();
            }
        }

        setupEventListeners() {
            // This will be implemented when parsing the template
        }

        render() {
            if (!this.el) return;

            // Parse template and render to DOM
            const html = this.parseTemplate(this.template);
            this.el.innerHTML = html;

            // Set up child components
            this.setupChildComponents();

            // Set up event listeners
            this.setupDOMEventListeners();

            // Mark as mounted
            this.isMounted = true;

            // Call mounted hook
            if (this.mounted) {
                this.mounted.call(this);
            }
        }

        parseTemplate(template) {
            // Simple template parsing - in a real implementation, this would be more complex
            let html = template;

            // Replace data bindings
            html = html.replace(/\{\{([^}]+)\}\}/g, (match, path) => {
                const value = this.state.get(path.trim());
                return value !== undefined ? value : '';
            });

            // Replace directives
            html = html.replace(/cx-(\w+)="([^"]+)"/g, (match, directive, value) => {
                return this.processDirective(directive, value);
            });

            return html;
        }

        processDirective(directive, value) {
            switch (directive) {
                case 'if':
                    return this.state.get(value) ? '' : 'style="display:none"';
                case 'show':
                    return this.state.get(value) ? '' : 'style="display:none"';
                case 'text':
                    return this.state.get(value) || '';
                case 'html':
                    return this.state.get(value) || '';
                case 'class':
                    return this.processClassDirective(value);
                case 'style':
                    return this.processStyleDirective(value);
                case 'on':
                    return this.processEventDirective(value);
                default:
                    return '';
            }
        }

        processClassDirective(value) {
            const classes = [];
            const conditions = value.split(',');
            
            for (const condition of conditions) {
                const [className, expression] = condition.trim().split(':');
                if (this.state.get(expression.trim())) {
                    classes.push(className.trim());
                }
            }
            
            return `class="${classes.join(' ')}"`;
        }

        processStyleDirective(value) {
            const styles = [];
            const properties = value.split(',');
            
            for (const property of properties) {
                const [prop, expression] = property.trim().split(':');
                styles.push(`${prop.trim()}: ${this.state.get(expression.trim())}`);
            }
            
            return `style="${styles.join('; ')}"`;
        }

        processEventDirective(value) {
            const [event, method] = value.split(':');
            return `on${event}="Webs.components['${this.name}'].methods.${method}(event)"`;
        }

        setupChildComponents() {
            // Find child component elements and initialize them
            const childElements = utils.find('[cx-component]', this.el);
            
            childElements.forEach(element => {
                const componentName = utils.getAttr(element, 'cx-component');
                const component = config.components[componentName];
                
                if (component) {
                    const child = new Component({
                        ...component,
                        el: element,
                        parent: this
                    });
                    
                    child.init();
                    this.children.push(child);
                }
            });
        }

        setupDOMEventListeners() {
            // Find elements with event handlers
            const eventElements = utils.find('[on\\w+]', this.el);
            
            eventElements.forEach(element => {
                const attributes = element.attributes;
                
                for (let i = 0; i < attributes.length; i++) {
                    const attr = attributes[i];
                    
                    if (attr.name.startsWith('on')) {
                        const event = attr.name.substring(2).toLowerCase();
                        const handler = attr.value;
                        
                        const listener = (e) => {
                            eval(`this.methods.${handler}(e)`);
                        };
                        
                        utils.on(element, event, listener.bind(this));
                        this.eventListeners.push({ element, event, listener });
                    }
                }
            });
        }

        destroy() {
            // Call beforeDestroy hook
            if (this.beforeDestroy) {
                this.beforeDestroy.call(this);
            }

            // Remove event listeners
            this.eventListeners.forEach(({ element, event, listener }) => {
                utils.off(element, event, listener);
            });

            // Destroy child components
            this.children.forEach(child => child.destroy());

            // Mark as unmounted
            this.isMounted = false;
        }

        $emit(eventName, ...args) {
            // Emit event to parent component
            if (this.parent) {
                this.parent.$emit(eventName, ...args);
            }
        }

        $on(eventName, callback) {
            // Listen for events from child components
            if (!this.eventListeners) {
                this.eventListeners = {};
            }
            
            if (!this.eventListeners[eventName]) {
                this.eventListeners[eventName] = [];
            }
            
            this.eventListeners[eventName].push(callback);
        }
    }

    // ===== ROUTER =====
    class Router {
        constructor(options = {}) {
            this.routes = options.routes || [];
            this.mode = options.mode || 'hash';
            this.base = options.base || '/';
            this.currentRoute = null;
            this.beforeEach = options.beforeEach || null;
            this.afterEach = options.afterEach || null;
            
            this.init();
        }

        init() {
            // Handle route changes
            if (this.mode === 'hash') {
                window.addEventListener('hashchange', this.handleRouteChange.bind(this));
                window.addEventListener('load', this.handleRouteChange.bind(this));
            } else {
                window.addEventListener('popstate', this.handleRouteChange.bind(this));
                window.addEventListener('load', this.handleRouteChange.bind(this));
                
                // Handle link clicks
                document.addEventListener('click', this.handleLinkClick.bind(this));
            }
            
            // Initial route
            this.handleRouteChange();
        }

        handleRouteChange() {
            const path = this.getCurrentPath();
            const route = this.matchRoute(path);
            
            if (route) {
                if (this.beforeEach) {
                    this.beforeEach(route, this.currentRoute);
                }
                
                this.currentRoute = route;
                this.renderRoute(route);
                
                if (this.afterEach) {
                    this.afterEach(route);
                }
            }
        }

        handleLinkClick(e) {
            const target = e.target.closest('a');
            
            if (!target) return;
            
            const href = target.getAttribute('href');
            
            if (href && href.startsWith('/') && !target.getAttribute('target')) {
                e.preventDefault();
                this.push(href);
            }
        }

        getCurrentPath() {
            if (this.mode === 'hash') {
                return window.location.hash.slice(1) || '/';
            } else {
                return window.location.pathname;
            }
        }

        matchRoute(path) {
            for (const route of this.routes) {
                const match = path.match(route.path);
                
                if (match) {
                    const params = {};
                    
                    if (route.paramNames) {
                        route.paramNames.forEach((name, index) => {
                            params[name] = match[index + 1];
                        });
                    }
                    
                    return {
                        ...route,
                        params,
                        path
                    };
                }
            }
            
            return null;
        }

        renderRoute(route) {
            if (route.component) {
                // Find the main app element
                const appElement = utils.findOne('#app') || document.body;
                
                // Clear previous content
                utils.empty(appElement);
                
                // Create component container
                const container = utils.create('div');
                appElement.appendChild(container);
                
                // Create and mount component
                const component = new Component({
                    ...route.component,
                    el: container,
                    props: {
                        ...route.params
                    }
                });
                
                component.init();
            }
        }

        push(path) {
            if (this.mode === 'hash') {
                window.location.hash = path;
            } else {
                window.history.pushState({}, '', path);
                this.handleRouteChange();
            }
        }

        replace(path) {
            if (this.mode === 'hash') {
                window.location.replace(`#${path}`);
            } else {
                window.history.replaceState({}, '', path);
                this.handleRouteChange();
            }
        }

        go(n) {
            window.history.go(n);
        }

        back() {
            this.go(-1);
        }

        forward() {
            this.go(1);
        }
    }

    // ===== DIRECTIVES =====
    const directives = {
        // Show/hide element based on condition
        'show': (el, binding) => {
            el.style.display = binding.value ? '' : 'none';
        },

        // Add/remove element from DOM based on condition
        'if': (el, binding) => {
            if (binding.value) {
                if (el.parentNode) {
                    el.style.display = '';
                }
            } else {
                el.style.display = 'none';
            }
        },

        // Set text content
        'text': (el, binding) => {
            el.textContent = binding.value;
        },

        // Set HTML content
        'html': (el, binding) => {
            el.innerHTML = binding.value;
        },

        // Toggle class
        'class': (el, binding) => {
            if (typeof binding.value === 'string') {
                utils.toggleClass(el, binding.value);
            } else if (typeof binding.value === 'object') {
                for (const [className, condition] of Object.entries(binding.value)) {
                    if (condition) {
                        utils.addClass(el, className);
                    } else {
                        utils.removeClass(el, className);
                    }
                }
            }
        },

        // Set style
        'style': (el, binding) => {
            if (typeof binding.value === 'string') {
                el.style.cssText = binding.value;
            } else if (typeof binding.value === 'object') {
                for (const [property, value] of Object.entries(binding.value)) {
                    el.style[property] = value;
                }
            }
        },

        // Form input binding
        'model': (el, binding, vnode) => {
            el.value = binding.value;
            
            utils.on(el, 'input', () => {
                binding.value = el.value;
            });
        },

        // Event handling
        'on': (el, binding, vnode) => {
            const [event, handler] = binding.arg.split(':');
            
            if (event && handler) {
                utils.on(el, event, (e) => {
                    vnode.context[handler](e);
                });
            }
        },

        // Lazy loading images
        'lazy': (el, binding) => {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        img.src = binding.value;
                        observer.unobserve(img);
                    }
                });
            });
            
            observer.observe(el);
        },

        // Infinite scroll
        'infinite': (el, binding) => {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        binding.value();
                    }
                });
            }, {
                rootMargin: '100px'
            });
            
            observer.observe(el);
        }
    };

    // ===== FILTERS =====
    const filters = {
        // Capitalize first letter
        'capitalize': (value) => {
            if (!value) return '';
            return value.charAt(0).toUpperCase() + value.slice(1);
        },

        // Uppercase
        'uppercase': (value) => {
            return value ? value.toString().toUpperCase() : '';
        },

        // Lowercase
        'lowercase': (value) => {
            return value ? value.toString().toLowerCase() : '';
        },

        // Currency formatting
        'currency': (value, symbol = '$', decimals = 2) => {
            const val = parseFloat(value);
            if (isNaN(val)) return '';
            return symbol + val.toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
        },

        // Percentage
        'percentage': (value, decimals = 0) => {
            const val = parseFloat(value);
            if (isNaN(val)) return '';
            return (val * 100).toFixed(decimals) + '%';
        },

        // Date formatting
        'date': (value, format = 'YYYY-MM-DD') => {
            return utils.formatDate(value, format);
        },

        // Pluralize
        'pluralize': (value, singular, plural) => {
            return value === 1 ? singular : plural;
        },

        // Truncate text
        'truncate': (value, length = 30, omission = '...') => {
            if (!value) return '';
            if (value.length <= length) return value;
            return value.substring(0, length) + omission;
        }
    };

    // ===== TRANSITIONS =====
    const transitions = {
        // Fade transition
        'fade': {
            enter(el, done) {
                utils.fadeIn(el, 300);
                setTimeout(done, 300);
            },
            leave(el, done) {
                utils.fadeOut(el, 300);
                setTimeout(done, 300);
            }
        },

        // Slide transition
        'slide': {
            enter(el, done) {
                utils.slideDown(el, 300);
                setTimeout(done, 300);
            },
            leave(el, done) {
                utils.slideUp(el, 300);
                setTimeout(done, 300);
            }
        },

        // Scale transition
        'scale': {
            enter(el, done) {
                el.style.transform = 'scale(0)';
                el.style.opacity = '0';
                el.style.transition = 'transform 0.3s, opacity 0.3s';
                
                setTimeout(() => {
                    el.style.transform = 'scale(1)';
                    el.style.opacity = '1';
                }, 10);
                
                setTimeout(done, 300);
            },
            leave(el, done) {
                el.style.transform = 'scale(0)';
                el.style.opacity = '0';
                
                setTimeout(done, 300);
            }
        }
    };

    // ===== PLUGINS =====
    const plugins = {
        // HTTP client
        'http': {
            install(Vue) {
                Vue.prototype.$http = {
                    get(url, options = {}) {
                        return utils.ajax({ ...options, method: 'GET', url });
                    },
                    post(url, data = {}, options = {}) {
                        return utils.ajax({
                            ...options,
                            method: 'POST',
                            url,
                            body: JSON.stringify(data)
                        });
                    },
                    put(url, data = {}, options = {}) {
                        return utils.ajax({
                            ...options,
                            method: 'PUT',
                            url,
                            body: JSON.stringify(data)
                        });
                    },
                    delete(url, options = {}) {
                        return utils.ajax({ ...options, method: 'DELETE', url });
                    }
                };
            }
        },

        // Storage
        'storage': {
            install(Vue) {
                Vue.prototype.$storage = {
                    set(key, value) {
                        if (typeof value === 'object') {
                            localStorage.setItem(key, JSON.stringify(value));
                        } else {
                            localStorage.setItem(key, value);
                        }
                    },
                    get(key) {
                        const value = localStorage.getItem(key);
                        try {
                            return JSON.parse(value);
                        } catch (e) {
                            return value;
                        }
                    },
                    remove(key) {
                        localStorage.removeItem(key);
                    },
                    clear() {
                        localStorage.clear();
                    }
                };
            }
        },

        // Event bus
        'eventBus': {
            install(Vue) {
                const eventBus = new Vue();
                Vue.prototype.$bus = eventBus;
            }
        }
    };

    // ===== MAIN Webs.JS OBJECT =====
    const Webs = {
        // Configuration
        config,

        // Utilities
        utils,

        // Core classes
        Reactive,
        Component,
        Router,

        // Directives
        directives,

        // Filters
        filters,

        // Transitions
        transitions,

        // Plugins
        plugins,

        // Component registry
        component(name, definition) {
            config.components[name] = definition;
        },

        // Directive registration
        directive(name, definition) {
            config.directives[name] = definition;
        },

        // Filter registration
        filter(name, definition) {
            config.filters[name] = definition;
        },

        // Transition registration
        transition(name, definition) {
            config.transitions[name] = definition;
        },

        // Plugin installation
        use(plugin, options = {}) {
            if (typeof plugin === 'function') {
                plugin(Webs, options);
            } else if (plugin.install) {
                plugin.install(Webs, options);
            }
            return Webs;
        },

        // Create app instance
        createApp(options = {}) {
            const app = {
                // Component registration
                component(name, definition) {
                    config.components[name] = definition;
                    return app;
                },

                // Directive registration
                directive(name, definition) {
                    config.directives[name] = definition;
                    return app;
                },

                // Filter registration
                filter(name, definition) {
                    config.filters[name] = definition;
                    return app;
                },

                // Plugin installation
                use(plugin, options = {}) {
                    Webs.use(plugin, options);
                    return app;
                },

                // Mount app
                mount(selector) {
                    const el = utils.findOne(selector);
                    
                    if (el) {
                        const component = new Component({
                            ...options,
                            el
                        });
                        
                        component.init();
                        return component;
                    }
                    
                    return null;
                }
            };

            return app;
        },

        // Initialize framework
        init() {
            // Auto-initialize components
            document.addEventListener('DOMContentLoaded', () => {
                const elements = utils.find('[cx-data]');
                
                elements.forEach(element => {
                    const data = utils.getAttr(element, 'cx-data');
                    
                    try {
                        const parsedData = JSON.parse(data);
                        const component = new Component({
                            data: parsedData,
                            el: element
                        });
                        
                        component.init();
                    } catch (e) {
                        console.error('Invalid cx-data JSON:', e);
                    }
                });
            });
        }
    };

    // Initialize framework
    Webs.init();

    // Expose to global
    window.Webs = Webs;
})();
/*EOF*/
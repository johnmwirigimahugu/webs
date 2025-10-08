# Webs Framework - Complete User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Routing System](#routing-system)
6. [Request & Response Handling](#request--response-handling)
7. [Template Engine](#template-engine)
8. [Database & ORM](#database--orm)
9. [Authentication System](#authentication-system)
10. [Middleware & Plugins](#middleware--plugins)
11. [Static Assets](#static-assets)
12. [Error Handling](#error-handling)
13. [CLI Tools](#cli-tools)
14. [Testing](#testing)
15. [Deployment](#deployment)
16. [Advanced Features](#advanced-features)
17. [Best Practices](#best-practices)
18. [Troubleshooting](#troubleshooting)

## Introduction

Webs is a zero-dependency Python web framework inspired by Flask but with additional features like built-in templating, ORM, authentication, and frontend assets. It aims for simplicity while providing enterprise-grade features.

### Key Features
- Zero external dependencies (pure Python)
- Built-in template engine with inheritance and macros
- Hybrid SQL/NoSQL ORM (RedBean-like)
- Authentication & authorization decorators
- CORS support
- Built-in CLI tools
- Hot reloading in development
- API documentation generation
- Security monitoring (Sentinel)

### Philosophy
"Simplicity is Superbly Sweet" - The framework prioritizes developer experience while maintaining powerful features under the hood.

## Installation & Setup

### Requirements
- Python 3.7+
- No external dependencies required

### Installation
```bash
# Download the framework file
wget https://your-repo/wabs.py

# Or clone the repository
git clone https://your-repo/webs-framework.git
```

### Project Structure
```
myproject/
├── app.py              # Main application
├── wabs.py             # Framework file
├── templates/          # Template directory
├── static/             # Static assets
├── webs.db            # SQLite database (auto-created)
└── requirements.txt    # Empty (zero dependencies!)
```

## Quick Start

### Hello World Application

```python
from webs import webs

# Create application instance
app = webs(__name__)

@app.route('/')
def home(request):
    return "Hello, World!"

@app.route('/user/<name>')
def user_profile(request, name):
    return f"Welcome, {name}!"

if __name__ == '__main__':
    app.run(debug=True)
```

### Running the Application
```bash
python app.py
# Server starts at http://127.0.0.1:8000
```

## Core Concepts

### Application Instance
The `webs` class is the central application object:

```python
from webs import webs

app = webs(__name__)  # __name__ helps with resource location
```

### Request Context
Every route handler receives a `Request` object:

```python
@app.route('/info')
def info(request):
    return f"Method: {request.method}, Path: {request.path}"
```

### Response Objects
Return responses in multiple formats:

```python
from webs import Response

@app.route('/json')
def json_response(request):
    return {"message": "Hello JSON"}  # Auto-converted

@app.route('/custom')
def custom_response(request):
    return Response("Custom", status_code=201, 
                   headers={'X-Custom': 'Value'})
```

## Routing System

### Basic Routing

```python
@app.route('/')
def index(request):
    return "Home page"

@app.route('/about', methods=['GET', 'POST'])
def about(request):
    if request.method == 'POST':
        return "Posted to about"
    return "About page"
```

### URL Parameters

```python
# String parameter (default)
@app.route('/user/<name>')
def user(request, name):
    return f"User: {name}"

# Integer parameter
@app.route('/post/<int:id>')
def post(request, id):
    return f"Post ID: {id} (type: {type(id)})"

# Path parameter (captures slashes)
@app.route('/files/<path:filename>')
def files(request, filename):
    return f"File: {filename}"

# Custom regex
@app.route('/custom/<re:[a-z]+:code>')
def custom(request, code):
    return f"Code: {code}"
```

### HTTP Methods

```python
@app.route('/api/data', methods=['GET', 'POST', 'PUT', 'DELETE'])
def api_data(request):
    if request.method == 'GET':
        return {"action": "read"}
    elif request.method == 'POST':
        return {"action": "create"}
    elif request.method == 'PUT':
        return {"action": "update"}
    elif request.method == 'DELETE':
        return {"action": "delete"}
```

## Request & Response Handling

### Request Object Properties

```python
@app.route('/request-info')
def request_info(request):
    info = {
        'method': request.method,
        'path': request.path,
        'query_string': request.query_string,
        'args': dict(request.args),  # URL parameters
        'headers': dict(request.headers),
        'cookies': request.cookies,
        'client_addr': request.client_addr,
        'content_type': request.content_type,
        'content_length': request.content_length
    }
    return info
```

### Handling Form Data

```python
@app.route('/form', methods=['GET', 'POST'])
def handle_form(request):
    if request.method == 'POST':
        if request.form:
            name = request.form.get('name', 'Anonymous')
            email = request.form.get('email')
            return f"Hello {name}, email: {email}"
        return "No form data received"
    
    return '''
    <form method="post">
        <input name="name" placeholder="Name">
        <input name="email" type="email" placeholder="Email">
        <button type="submit">Submit</button>
    </form>
    '''
```

### Handling JSON Data

```python
@app.route('/api/users', methods=['POST'])
def create_user(request):
    if request.json:
        user_data = request.json
        # Process user data
        return {"status": "created", "user": user_data}
    return {"error": "JSON data required"}, 400
```

### Response Types

```python
@app.route('/responses')
def response_examples(request):
    # String response (auto-converted to Response)
    return "Simple text"

@app.route('/json-resp')
def json_response(request):
    # Dict/List auto-converted to JSON
    return {"key": "value", "list": [1, 2, 3]}

@app.route('/custom-resp')
def custom_response(request):
    # Custom Response object
    resp = Response("Custom content", status_code=201)
    resp.headers['X-Custom-Header'] = 'Custom Value'
    return resp

@app.route('/redirect-resp')
def redirect_response(request):
    return Response.redirect('/other-page')

@app.route('/file-resp')
def file_response(request):
    return Response.send_file('static/image.png')
```

### Cookies

```python
@app.route('/set-cookie')
def set_cookie(request):
    resp = Response("Cookie set")
    resp.set_cookie('username', 'john', max_age=3600, http_only=True)
    return resp

@app.route('/get-cookie')
def get_cookie(request):
    username = request.cookies.get('username', 'Guest')
    return f"Hello {username}"

@app.route('/delete-cookie')
def delete_cookie(request):
    resp = Response("Cookie deleted")
    resp.delete_cookie('username')
    return resp
```

## Template Engine

### Basic Templates

Create `templates/hello.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
</head>
<body>
    <h1>Hello {{ name }}!</h1>
    <p>Today is {{ date }}</p>
</body>
</html>
```

Render in your route:
```python
from webs import render_template

@app.route('/hello/<name>')
def hello(request, name):
    context = {
        'title': 'Greeting Page',
        'name': name,
        'date': '2025-09-20'
    }
    return render_template('hello.html', **context)
```

### Template Inheritance

Base template `templates/base.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}Default Title{% endblock %}</title>
    {{ webs_assets.inject() | safe }}
</head>
<body>
    <nav>
        <a href="/">Home</a>
        <a href="/about">About</a>
    </nav>
    
    <main>
        {% block content %}{% endblock %}
    </main>
    
    <footer>
        {% block footer %}© 2025 My App{% endblock %}
    </footer>
</body>
</html>
```

Child template `templates/page.html`:
```html
{% extends "base.html" %}

{% block title %}{{ page_title }}{% endblock %}

{% block content %}
    <h1>{{ heading }}</h1>
    <p>{{ content }}</p>
{% endblock %}
```

### Template Macros

Define reusable components:
```html
{% macro render_field(field) %}
    <div class="field">
        <label>{{ field.label }}</label>
        <input type="{{ field.type }}" name="{{ field.name }}" value="{{ field.value }}">
        {% if field.error %}
            <span class="error">{{ field.error }}</span>
        {% endif %}
    </div>
{% endmacro %}

<!-- Usage -->
{{ render_field(name_field) }}
{{ render_field(email_field) }}
```

### Template Filters

Built-in filters:
```html
{{ name | upper }}           <!-- Convert to uppercase -->
{{ content | lower }}        <!-- Convert to lowercase -->
{{ html_content | safe }}    <!-- Don't escape HTML -->
{{ user_input | e }}         <!-- Escape HTML (default) -->
```

Custom filters:
```python
def reverse_filter(text):
    return text[::-1]

app.template_engine.add_filter('reverse', reverse_filter)
```

Use in templates:
```html
{{ "hello world" | reverse }}  <!-- outputs: dlrow olleh -->
```

### Advanced Template Engine

```python
# Configure template engine
app = webs(__name__)
app.template_engine = websTemplateEngine(
    template_dir="templates",
    autoescape=True,      # Auto-escape HTML
    sandboxed=True,       # Security sandbox
    cache_size=100        # Template cache size
)

# Custom template extension
class MyExtension(TemplateExtension):
    def filter_markdown(self, text):
        # Convert markdown to HTML
        return text.replace('**', '<b>').replace('**', '</b>')
    
    def tag_cache(self, parser, tag_name, args):
        # Custom {% cache %} tag implementation
        pass

app.template_engine.add_extension(MyExtension)
```

## Database & ORM

### Model Definition

```python
from webs import Model, Field

class User(Model):
    name = Field(str, nullable=False)
    email = Field(str, unique=True)
    age = Field(int, default=0)
    is_active = Field(bool, default=True)

class Post(Model):
    title = Field(str, nullable=False)
    content = Field(str)
    user_id = Field(int)
    created_at = Field(str)  # You can store datetime as string
```

### Basic CRUD Operations

```python
@app.route('/users', methods=['POST'])
def create_user(request):
    # Create new user
    user = User(
        name=request.form.get('name'),
        email=request.form.get('email'),
        age=int(request.form.get('age', 0))
    )
    user.save(app)
    return {"message": "User created", "id": user.id}

@app.route('/users')
def list_users(request):
    # Find all users (you'll need to implement this)
    users = User.all(app)  # This method needs to be added
    return {"users": [user.__dict__ for user in users]}

@app.route('/users/<int:user_id>')
def get_user(request, user_id):
    # Find user by ID
    users = User.find(app, id=user_id)
    if users:
        return {"user": users[0].__dict__}
    return {"error": "User not found"}, 404

@app.route('/users/<int:user_id>', methods=['PUT'])
def update_user(request, user_id):
    users = User.find(app, id=user_id)
    if users:
        user = users[0]
        user.name = request.json.get('name', user.name)
        user.email = request.json.get('email', user.email)
        user.save(app)
        return {"message": "User updated"}
    return {"error": "User not found"}, 404
```

### Database Configuration

```python
# Freeze ORM to prevent automatic table creation in production
app.freeze_orm()

# Custom database connection (if needed)
import sqlite3
app._orm_conn = sqlite3.connect('custom.db')
```

## Authentication System

### User Authentication

```python
from webs import login_user, logout_user, current_user, login_required

@app.route('/login', methods=['GET', 'POST'])
def login(request):
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Verify credentials (implement your logic)
        user = authenticate_user(email, password)
        if user:
            login_user(request, user)
            return Response.redirect('/dashboard')
        else:
            return "Invalid credentials", 401
    
    return '''
    <form method="post">
        <input name="email" type="email" placeholder="Email" required>
        <input name="password" type="password" placeholder="Password" required>
        <button type="submit">Login</button>
    </form>
    '''

@app.route('/logout')
def logout(request):
    logout_user(request)
    return Response.redirect('/')

@app.route('/profile')
@login_required
def profile(request):
    user = current_user(request)
    return f"Welcome, {user['name']}!"
```

### Role-Based Access Control

```python
from webs import roles_required

@app.route('/admin')
@roles_required('admin')
def admin_panel(request):
    return "Admin only content"

@app.route('/moderator')
@roles_required('admin', 'moderator')
def moderator_panel(request):
    return "Admin or moderator content"

@app.route('/user-only')
@roles_required('user')
def user_content(request):
    return "Regular user content"
```

### Guest-Only Pages

```python
from webs import anonymous_only

@app.route('/register')
@anonymous_only
def register(request):
    # Only accessible to non-authenticated users
    return "Registration form"
```

## Middleware & Plugins

### Custom Middleware

```python
def logging_middleware(request):
    print(f"Request: {request.method} {request.path}")
    return request  # Must return request object

def auth_middleware(request):
    # Add user to request if authenticated
    user = current_user(request)
    request.g.user = user
    return request

# Register middleware
app.use(logging_middleware)
app.use(auth_middleware)
```

### Before/After Request Handlers

```python
@app.before_request
def before_each_request(request):
    # Runs before every request
    request.g.start_time = time.time()
    # Return Response to short-circuit request

@app.after_request
def after_each_request(request, response):
    # Runs after every request
    duration = time.time() - request.g.start_time
    response.headers['X-Response-Time'] = f"{duration:.3f}s"
    return response
```

### CORS Support

```python
from webs import CORS

# Enable CORS for all routes
cors = CORS(app, 
           allowed_origins=['http://localhost:3000', 'https://mydomain.com'],
           allowed_methods=['GET', 'POST', 'PUT', 'DELETE'],
           allowed_headers=['Content-Type', 'Authorization'],
           allow_credentials=True)
```

### Custom Plugins

```python
class DatabasePlugin:
    def __init__(self, db_url):
        self.db_url = db_url
    
    def install(self, app):
        # Initialize database connection
        app.db = self.create_connection()
    
    def create_connection(self):
        # Database setup logic
        pass

# Install plugin
db_plugin = DatabasePlugin('sqlite:///app.db')
app.add_plugin(db_plugin)
```

## Static Assets

### Built-in Assets

The framework includes built-in web assets:

```python
from webs import websWebAssets

# Register asset routes
websWebAssets.register_routes(app)

# In templates, inject assets
# {{ webs_assets.inject() | safe }}
```

This provides:
- `/static/w3.css` - W3.CSS framework
- `/static/w3.js` - W3.JS utilities  
- `/static/ahah.js` - AHAH (Asynchronous HTML and HTTP)
- `/static/brython.js` - Python in the browser

### Custom Static Files

```python
@app.route('/static/<path:filename>')
def static_files(request, filename):
    return Response.send_file(f"static/{filename}")

# Serve with caching
@app.route('/assets/<path:filename>')
def cached_assets(request, filename):
    return Response.send_file(
        f"assets/{filename}",
        max_age=3600  # Cache for 1 hour
    )
```

## Error Handling

### Custom Error Pages

```python
@app.errorhandler(404)
def not_found(request, error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def server_error(request, error):
    return render_template('errors/500.html'), 500

@app.errorhandler(Exception)
def handle_exception(request, error):
    # Log error
    app.sentinel.log('ERROR', str(error))
    return "Something went wrong", 500
```

### HTTP Exceptions

```python
from webs import HTTPException

@app.route('/restricted')
def restricted(request):
    if not user_has_permission():
        raise HTTPException(403, "Access denied")
    return "Secret content"
```

## CLI Tools

### Built-in Commands

```bash
# Start new project
python webs.py startproject myapp

# Run development server
python webs.py run

# Run tests
python webs.py test
```

### Custom CLI Commands

```python
@app.cli.command()
def migrate():
    """Run database migrations"""
    print("Running migrations...")
    # Migration logic here

@app.cli.command()
def seed():
    """Seed database with sample data"""
    # Create sample users
    for i in range(10):
        user = User(name=f"User {i}", email=f"user{i}@example.com")
        user.save(app)
    print("Database seeded!")
```

## Testing

### Test Framework

```python
from webs import test

@test
def test_home_page():
    """Test home page returns 200"""
    # Create test client
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert "Hello" in response.data

@test
def test_user_creation():
    """Test user model"""
    user = User(name="Test User", email="test@example.com")
    user.save(app)
    found_users = User.find(app, email="test@example.com")
    assert len(found_users) == 1
    assert found_users[0].name == "Test User"
```

### Running Tests

```bash
python webs.py test
```

## Deployment

### Production Configuration

```python
# production.py
from webs import webs
import os

app = webs(__name__)

# Production settings
app.config = {
    'DEBUG': False,
    'SECRET_KEY': os.environ.get('SECRET_KEY'),
    'DATABASE_URL': os.environ.get('DATABASE_URL')
}

# Freeze ORM in production
app.freeze_orm()

# Register production middleware
@app.before_request
def security_headers(request):
    # Add security headers
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
```

### WSGI Deployment

```python
# For deployment with Gunicorn, uWSGI, etc.
from webs import webs

app = webs(__name__)
# ... your routes ...

# WSGI entry point
application = app.wsgi_app

# Run with: gunicorn app:application
```

### ASGI Deployment (Future)

```python
# For deployment with Uvicorn, Hypercorn (when async support is added)
application = app.asgi_app

# Run with: uvicorn app:application
```

## Advanced Features

### API Documentation

```python
@app.api('/users', methods=['GET'], 
         summary='List all users',
         description='Returns a paginated list of all users')
def api_list_users(request):
    return {"users": []}

@app.api('/users', methods=['POST'],
         summary='Create user',
         description='Create a new user account')
def api_create_user(request):
    return {"message": "Created"}

# Auto-generated docs at /docs
```

### Server-Sent Events (SSE)

```python
from webs import SSE

@app.route('/events')
async def events(request):
    response = Response(content_type='text/event-stream')
    
    for i in range(10):
        await SSE.send(response, 'update', {'count': i})
        await asyncio.sleep(1)
    
    return response
```

### WebSocket Support (Planned)

```python
@app.websocket('/ws')
async def websocket_handler(websocket):
    while True:
        data = await websocket.receive_json()
        await websocket.send_json({"echo": data})
```

### Background Tasks

```python
import threading

def background_task():
    # Long-running task
    time.sleep(60)
    print("Background task completed")

@app.route('/start-task')
def start_task(request):
    thread = threading.Thread(target=background_task)
    thread.daemon = True
    thread.start()
    return {"message": "Task started"}
```

## Best Practices

### Project Structure

```
myapp/
├── app.py                  # Main application
├── webs.py                 # Framework file
├── models/                 # Database models
│   ├── __init__.py
│   ├── user.py
│   └── post.py
├── views/                  # Route handlers
│   ├── __init__.py
│   ├── auth.py
│   └── api.py
├── templates/              # Jinja2 templates
│   ├── base.html
│   ├── home.html
│   └── auth/
│       ├── login.html
│       └── register.html
├── static/                 # Static assets
│   ├── css/
│   ├── js/
│   └── images/
├── tests/                  # Test files
│   ├── test_models.py
│   └── test_views.py
├── config.py              # Configuration
└── requirements.txt       # Empty (zero deps!)
```

### Configuration Management

```python
# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    DEBUG = False

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

# app.py
from config import DevelopmentConfig

app = webs(__name__)
app.config.from_object(DevelopmentConfig)
```

### Security Best Practices

```python
# Always use HTTPS in production
# Set secure session cookies
# Implement CSRF protection
# Validate all user input
# Use parameterized queries (ORM does this)
# Enable security headers

@app.before_request
def security_headers(request):
    # Add security headers to all responses
    pass

@app.after_request
def add_security_headers(request, response):
    response.headers.update({
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    })
    return response
```

### Performance Tips

```python
# Use template caching in production
app.template_engine.cache_size = 1000

# Freeze ORM in production
app.freeze_orm()

# Minimize database queries
# Use proper indexing
# Implement response caching
# Optimize static asset delivery
```

## Troubleshooting

### Common Issues

**1. Template Not Found**
```
TemplateError: Template 'template.html' not found
```
- Check template directory path
- Ensure file exists and has correct permissions
- Verify template name spelling

**2. Database Connection Issues**
```
sqlite3.OperationalError: database is locked
```
- Close existing database connections
- Check file permissions
- Ensure only one process accesses database

**3. Import Errors**
```python
# Circular import issues
# Solution: Import at function level or restructure
def my_view(request):
    from .models import User  # Import inside function
    return User.find(app, id=1)
```

**4. Memory Issues with Large Responses**
```python
# Use streaming for large files
@app.route('/large-file')
def large_file(request):
    def generate():
        with open('large_file.txt', 'rb') as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                yield chunk
    
    return Response(generate(), content_type='application/octet-stream')
```

### Debug Mode

```python
# Enable debug mode for development
app.run(debug=True)

# This enables:
# - Hot reloading on code changes
# - Detailed error pages
# - Request logging
```

### Logging

```python
# Use built-in Sentinel for monitoring
app.sentinel.log('INFO', 'Application started')
app.sentinel.log('ERROR', f'Database error: {error}')

# Check logs
for log in app.sentinel.logs:
    print(f"[{log['level']}] {log['message']}")
```

---

## Framework Limitations & Considerations

While reviewing the framework code, I notice several areas that need attention:

### Missing Implementations
1. **Session Store**: The `_InMemorySessionStore` class is referenced but not implemented
2. **AsyncBytesIO**: Referenced in Request class but not defined
3. **Template AST Processing**: The template engine has placeholder methods
4. **WebSocket Support**: Mentioned but not implemented
5. **Request.from_wsgi/from_asgi**: Referenced but not implemented

### Architecture Concerns
1. **HTTP Server**: Uses basic `HTTPServer` which may not handle concurrent requests well
2. **Database Connections**: No connection pooling or thread safety
3. **Session Security**: No session encryption or CSRF protection
4. **Template Security**: Sandbox validation is incomplete

### Recommended Enhancements
1. Implement proper async support throughout
2. Add comprehensive error handling
3. Implement session management
4. Complete template engine features
5. Add test coverage
6. Improve documentation

The framework shows promise but needs significant development to be production-ready. Consider it a solid foundation that requires additional implementation work.

---

*This manual covers the intended functionality of the Webs framework. Some features may require additional implementation to work as described.*
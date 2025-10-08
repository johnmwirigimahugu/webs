# webs
🌐Webs (wabseth-5000): The Zero-Dependency Python Full-Stack Framework. Maximal security, minimal code. Features include PBKDF2-hashed passwords, sandboxed templates, thread-safe ORM, and WSGI adapter. Deployable instantly to webs.pythonanywhere.com.

# Webs.py (wabs-5000) Full-Stack Framework Documentation

## 1. Architectural Overview

### The Webs Application Class: Service Locator and Front Controller

The `Webs` class is the heart of the framework, serving two primary design patterns: the **Service Locator** and the **Front Controller**. As a Service Locator, it acts as a central registry for application-wide services and configurations. Developers register routes, configure database connections, initialize the template engine, and set up session stores through a single `app` instance. This centralization simplifies dependency management and provides a clear, single point of truth for the application's configuration.

As a **Front Controller**, the `Webs` class provides a single entry point for all incoming HTTP requests. Instead of having multiple scripts handling different URLs, the web server (e.g., `ThreadingHTTPServer` or a WSGI server) is configured to pass every request to the `Webs` application. The framework then takes responsibility for dispatching the request to the appropriate handler function based on the defined URL patterns.

### Request-Response Lifecycle

The framework operates on a clear and efficient pipeline, transforming a raw HTTP request into a processed response:

1.  **Receive:** The HTTP server receives a raw HTTP request from a client.
2.  **Handler Instantiation:** The server instantiates `WebsHTTPRequestHandler`, passing the `Webs` application instance to it.
3.  **Request Object Creation:** The handler parses the incoming request data (method, path, headers, body) and encapsulates it into a unified `Request` object. This object provides convenient, lazy-loaded properties like `.args`, `.form`, and `.json`.
4.  **Dispatch:** The handler calls the `app.dispatch_request(request)` method. The `Webs` application iterates through its registered routes, using the `URLPatternRouter` to find a match for the request path.
5.  **Execution:** Once a matching route is found, the associated view function is called, receiving the `Request` object as its primary argument.
6.  **Response Object Creation:** The view function performs its logic (database queries, template rendering, etc.) and returns a `Response` object, which encapsulates the status code, headers, and body.
7.  **Transmit:** The `WebsHTTPRequestHandler` takes the `Response` object, formats it according to the HTTP protocol, and transmits it back to the client. Session data is automatically saved at this stage if it has been modified.

## 2. Core Components: HTTP Handling

### The Request Object

The `Request` object is a rich, dictionary-like abstraction over the incoming HTTP request. It provides developers with intuitive access to all parts of the request without having to parse raw headers or query strings manually.

*   **`.args`**: A `MultiDict` containing the parsed URL query parameters (e.g., `/search?q=python`).
*   **`.form`**: A `MultiDict` containing parsed form data from `POST` or `PUT` requests with `application/x-www-form-urlencoded` content type. This is lazy-loaded for efficiency.
*   **`.json`**: A dictionary containing the parsed JSON body from a request with `application/json` content type. Also lazy-loaded.
*   **`.headers`**: A case-insensitive `NoCaseDict` for accessing request headers.
*   **`.cookies`**: A standard dictionary for accessing cookies sent by the client.
*   **`.session`**: A `SessionDict` object for interacting with user session data.

### The Response Object

The `Response` object is the primary mechanism for sending data back to the client. It is designed to be flexible, automatically handling content types and encoding for various data types.

*   **Constructor**: `Response(body='', status_code=200, headers=None)`. It intelligently handles `body` as `str`, `bytes`, or `dict`/`list` (automatically serializing to JSON).
*   **Factory Methods**: For convenience and clarity, the `Response` class provides several factory methods:
    *   `Response.json(data, status_code=200)`: Creates a response with `Content-Type: application/json`.
    *   `Response.redirect(location, status_code=302)`: Creates a redirect response, setting the `Location` header.
*   **Cookie Management**: Provides `set_cookie()` and `delete_cookie()` methods with full support for attributes like `max_age`, `path`, `domain`, `secure`, and `httponly`.

### URLPatternRouter: Dynamic and Type-Safe Routing

The `URLPatternRouter` is responsible for mapping incoming request paths to the correct view function. It uses the standard library's **`re` module** to compile URL patterns into efficient regular expressions, enabling powerful and type-safe dynamic routes.

*   **Static Routes**: Simple strings like `/about` are matched directly.
*   **Dynamic Routes**: Patterns can capture variables from the URL.
    *   `<name>`: Captures a string segment (e.g., `/user/<username>`).
    *   `<int:id>`: Captures an integer and automatically converts it to the `int` type (e.g., `/post/<int:id>`).
    *   `<float:price>`: Captures a floating-point number.
    *   `<path:subpath>`: Captures a path, including slashes, useful for nested resources (e.g., `/files/<path:filepath>`).

This type conversion eliminates the need for manual casting in view functions and provides a layer of validation, returning a 404 error if the type conversion fails.

## 3. Data Layer: ORM and Persistence

### DatabasePool: Thread-Safe Connection Management

To ensure efficient and safe database access in a multi-threaded environment, the framework provides a `DatabasePool`. This is a critical component for performance, as it avoids the overhead of establishing a new database connection for every request. Its implementation relies solely on standard library concurrency tools.

*   **Mechanism**: The pool uses a **`collections.deque`** to act as a stack of available database connections.
*   **Thread Safety**: Access to the deque and the set of used connections is protected by a **`threading.Lock`**. This ensures that only one thread can modify the pool's state at a time, preventing race conditions.
*   **SQLite3 Focus**: By default, the pool is configured for **SQLite3**, reinforcing the framework's zero-dependency philosophy. The `_create_connection` method establishes a connection with `check_same_thread=False`, as the pool itself manages thread safety.

### The Model ORM: Active Record Pattern

The `Model` class provides a lightweight, intuitive Object-Relational Mapping (ORM) layer following the Active Record pattern. Each model subclass represents a database table, and instances of that class represent rows in the table.

*   **Schema Definition**: Table schemas are defined declaratively using `Field` objects (`Field(int)`, `Field(str)`, etc.). A metaclass (`ModelMeta`) introspects these class attributes to build an understanding of the table structure.
*   **CRUD Operations**: Models provide simple methods for database interaction:
    *   `save(app)`: Inserts a new record or updates an existing one.
    *   `delete(app)`: Deletes the record from the database.
    *   `find(app, **conditions)`: A class method to find multiple records based on conditions.
    *   `find_one(app, **conditions)`: Finds a single record.
    *   `all(app)`: Retrieves all records from the table.
*   **"Frozen" Mode**: For production hardening, the ORM can be put into a "frozen" mode (`app._orm_frozen = True`), which prevents the creation of new tables, protecting against accidental schema modifications.

## 4. Presentation Layer: Templating and Sandboxing

### The websTemplateEngine

The `websTemplateEngine` is a secure and performant template engine designed to separate presentation logic from application code. It compiles template files into executable Python code for fast rendering.

### Template Syntax and Rendering

The syntax is familiar and concise, borrowing from popular modern engines:
*   `{{ variable }}`: Outputs a variable. By default, the output is HTML-escaped to prevent XSS attacks.
*   `{% python_code %}`: Allows for embedding Python logic like loops, conditionals, and variable assignments.
*   `{{ variable|filter }}`: Applies a filter to the variable (e.g., `{{ name|title }}`).

### Security Sandboxing and AST Analysis

Security is the foremost concern in the template engine. It is designed to prevent **Remote Code Execution (RCE)** by strictly controlling the execution environment.

*   **Code Transformation**: The template is first transformed into a Python code string. This process replaces `{{...}}` and `{%...%}` blocks with Python statements.
*   **Sandboxed Execution Context**: Before the generated code is executed using the built-in **`compile()`** function, it is run within a heavily restricted global context. This context is a dictionary that explicitly removes dangerous built-in functions and modules such as `open`, `exec`, `eval`, `import`, `__import__`, and others.
*   **AST Analysis (Conceptual)**: While the current implementation uses string transformation and a restricted globals dictionary, the design is conceptually aligned with **AST Analysis**. The `TemplateASTTransformer` class is structured to allow for future enhancement to a full Abstract Syntax Tree (AST) manipulation, which would provide even more robust static analysis and security guarantees by directly inspecting and modifying the code's structure before execution.

## 5. Security Layer: Authentication and Cryptography

### The SecurityManager

The `SecurityManager` is a centralized utility class that provides implementations of common security primitives, ensuring they are used correctly and consistently throughout the application. It relies exclusively on Python's standard cryptography modules.

### Password Hashing with PBKDF2-HMAC-SHA256

Storing plain-text passwords is a critical security vulnerability. The framework enforces secure password storage using a modern, key-stretching algorithm.

*   **Algorithm**: It uses the **PBKDF2-HMAC-SHA256** algorithm, implemented via the `hashlib.pbkdf2_hmac` function.
*   **Process**:
    1.  A cryptographically secure random salt is generated for each password using `os.urandom(32)`.
    2.  The password is hashed with the salt over 100,000 iterations. This high iteration count makes brute-force or rainbow table attacks computationally infeasible.
    3.  Both the resulting hash and the unique salt are stored in the database.
*   **Verification**: To verify a password, the process is repeated with the stored salt, and the resulting hash is compared to the stored hash.

### CSRF Protection

Cross-Site Request Forgery (CSRF) attacks are mitigated using the standard double-submit cookie pattern, implemented with constant-time comparison to prevent timing attacks.

*   **Token Generation**: Upon user login or session start, a unique, random token is generated using `secrets.token_urlsafe(32)`. This token is stored in the user's server-side session.
*   **Token Injection**: The token must be included in any state-changing form as a hidden input field.
*   **Verification**: When the form is submitted, the framework compares the token from the request body with the token in the session. The comparison is performed using **`hmac.compare_digest`**, which is a constant-time comparison function. This prevents an attacker from being able to guess the token character by character based on response times, neutralizing timing attacks.

## 6. Deployment: Production Readiness

### The WSGIAdapter for Universal Deployment

To ensure the framework can run on any production-grade Python web server, it includes a `WSGIAdapter`. This adapter acts as a bridge between the WSGI standard and the framework's internal `Request`/`Response` objects.

*   **WSGI Standard**: The Web Server Gateway Interface (WSGI) is a PEP 333 standard that defines a simple and universal interface between web servers and Python web applications or frameworks.
*   **Adapter Functionality**: The `WSGIAdapter` is a callable that receives the WSGI `environ` dictionary and `start_response` callable. It performs the following translations:
    1.  It parses the `environ` dictionary to construct the framework's native `Request` object.
    2.  It dispatches this request to the `Webs` application.
    3.  It receives the `Response` object back from the application.
    4.  It calls `start_response` with the status code and headers from the `Response` object.
    5.  It returns an iterable of the response body, as required by the WSGI standard.

This allows a `wabs-5000` application to be deployed on any WSGI server, including Gunicorn, uWSGI, or mod_wsgi for Apache.

### Deployment Example: PythonAnywhere

PythonAnywhere is a popular PaaS (Platform as a Service) that uses a WSGI-compatible server to run Python web applications. Deploying the `wabs-5000` framework on a platform like PythonAnywhere is straightforward due to the `WSGIAdapter`.

To deploy an application at the domain **`webs.pythonanywhere.com`**, one would:
1.  Upload the project files, including `webs.py` and the application code (`app.py`).
2.  In the PythonAnywhere web tab, configure the application to use a manual WSGI configuration.
3.  Edit the WSGI configuration file (e.g., `var/www/webs_pythonanywhere_com_wsgi.py`) to contain the following code:

```python
import sys
# add your project directory to the path
project_home = '/path/to/your/project'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

from webs import WSGIAdapter
from app import app # import your Webs app instance

# This line connects the WSGI server to the Webs framework
application = WSGIAdapter(app)

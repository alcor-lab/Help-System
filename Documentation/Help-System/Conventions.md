# Conventions
## C++ Naming Convention

```
| Kind                    | Convention             | Examples        |
| ----------------------- | ---------------------- | --------------- |
| Types and classes       | PascalCase             | MyClass, MyType |
| Variables and functions | camelCase              | myVariable      |
| Namespaces              | snake_case             | my_namespace    |
| Class members           | "m_" ++ camelCase      | m_myVariable    |
| Static class members    | "s_" ++ camelCase      | s_myStaticVar   |
| Indexes                 | "i" ++ PascalCase      | mat[iRow][iCol] |
| To avoid keywords       | name + "_"             | alignas_        |
```

Rules:
* Control structures (for, if, etc.) should have complete curly-braced block of code.
* Curly brackets from the same pair should be either in the same line or in the same column.
* Unnamed namespaces are not allowed in header files (use namespace "detail" instead).
* Expressions "using namespace" are not allowed in header files.

Tips:

* Use the *Case Convertion* package for Sublime Text.

## Python Naming Convention

```
| Kind                    | Convention             | Examples        |
| ----------------------- | ---------------------- | --------------- |
| Types and classes       | PascalCase             | MyClass, MyType |
| Variables and functions | snake_case             | my_variable     |
| Libraries               | snake_case             | my_module       |
| To avoid keywords       | name + "_"             | id_             |
```

In general, please follow the [PEP 8 style guide](https://www.python.org/dev/peps/pep-0008/).

## C++ list of keywords

Reference: http://en.cppreference.com/w/cpp/keyword

- alignas
- alignof
- and
- and_eq
- asm
- atomic_cancel
- atomic_commit
- atomic_noexcept
- auto
- bitand
- bitor
- bool
- break
- case
- catch
- char
- char16_t
- char32_t
- class
- compl
- concept
- const
- const_cast
- constexpr
- continue
- decltype
- default
- delete
- do
- double
- dynamic_cast
- else
- enum
- explicit
- export
- extern
- false
- final
- float
- for
- friend
- goto
- if
- import
- inline
- int
- long
- module
- mutable
- namespace
- new
- noexcept
- not
- not_eq
- nullptr
- operator
- or
- or_eq
- override
- private
- protected
- public
- register
- reinterpret_cast
- requires
- return
- short
- signed
- sizeof
- static
- static_assert
- static_cast
- struct
- switch
- synchronized
- template
- this
- thread_local
- throw
- transaction_safe
- transaction_safe_dynamic
- true
- try
- typedef
- typeid
- typename
- union
- unsigned
- using
- virtual
- void
- volatile
- wchar_t
- while
- xor
- xor_eq

## Python list of keywords

Reference: https://docs.python.org/3/reference/lexical_analysis.html

- False
- None
- True
- and
- as
- assert
- break
- class
- continue
- def
- del
- elif
- else
- except
- finally
- for
- from
- global
- if
- import
- in
- is
- lambda
- nonlocal
- not
- or
- pass
- raise
- return
- try
- while
- with
- yield

## Python list of builtins

Reference: https://docs.python.org/3/library/functions.html

- `__import__`
- abs
- all
- any
- ascii
- bin
- bool
- bytearray
- bytes
- callable
- chr
- classmethod
- compile
- complex
- delattr
- dict
- dir
- divmod
- enumerate
- eval
- exec
- filter
- float
- format
- frozenset
- getattr
- globals
- hasattr
- hash
- help
- hex
- id
- input
- int
- isinstance
- issubclass
- iter
- len
- list
- locals
- map
- max
- memoryview
- min
- next
- object
- oct
- open
- ord
- pow
- print
- property
- range
- repr
- reversed
- round
- set
- setattr
- slice
- sorted
- staticmethod
- str
- sum
- super
- tuple
- type
- vars
- zip

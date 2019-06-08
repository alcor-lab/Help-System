# How to contribute

1. Create a new branch based on the `master` branch.
2. Prepend it with `bugfix/` or `feature/` depending on the nature of the contributions. Example: `bugfix/wrong-title`. Choose a different prefix if it's justified.
3. Make sure it is formatted according to `.clang-format`. Use: 

`./Scripts/Help-System/setup.py run --clang-format`

4. Make sure it passes the clang-tidy check configured with `.clang-tidy`. Use:

`./Scripts/Help-System/setup.py run --clang-tidy`

4. Make sure that it passes all the tests:

```
./Scripts/Help-System/setup.py build --tests && ./Scripts/Help-System/setup.py run --tests
```

4. Once you fix any issues that were uncovered, push it to your branch:

```
git push origin your_branch
```

5. Open a pull request against the project's `master` branch.

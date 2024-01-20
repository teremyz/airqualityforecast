poetry_install:
	pipx install poetry
	echo 'export PATH="$HOME/.poetry/bin:$PATH"' >> ~/.bashrc

pyenv_install:
	sudo apt-get update
	sudo apt-get install -y \
		libbz2-dev \
		libreadline-dev \
		libssl-dev \
		libsqlite3-dev \
		libncurses5-dev \
		libncursesw5-dev \
		libffi-dev \
		zlib1g-dev \
		liblzma-dev \
		uuid-dev \
		tk-dev
	@echo "Install pyenv..."
	curl https://pyenv.run | bash
	echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
	echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
	echo '"$(pyenv init -)' >> ~/.bashrc

poetry_setup:
	# Poetry creates project dirm it most be empty
	poetry new AirQualityForecast
	cp Makefile AirQualityForecast/.

project_setup:
	# Go to project folder and run this
	mkdir -p src notebooks core scripts
	pyenv local 3.10.10

	echo "poetry init"
	poetry env use python
	poetry add --dev pytest-cov pre-commit flake8 mypy isort
	poetry add --dev --allow-prereleases black
	poetry add --dev ipykernel

	echo "Git init"
	echo '.coverage' > .gitignore
	echo '.vscode/\n.idea/' >> .gitignore
	curl -s https://raw.githubusercontent.com/github/gitignore/master/Python.gitignore >> .gitignore
	git init
	git add .
	git commit -m "init"
	git branch -M main
	git remote add origin https://github.com/teremyz/airqualityforecast.git
	git push -u origin main

	echo "pre-commit init"
	pre-commit install
	pre-commit autoupdate
	pre-commit run --all-files
	pre-commit run --all-files

poetry_shell:
	poetry shell
	code .

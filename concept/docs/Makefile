# Default target which builds the documentation and prepares it for publication
publish: html
	@mkdir -p $(PUBLISHDIR)
	@cp -r $(BUILDDIR)/html/* $(PUBLISHDIR)/
	@touch $(PUBLISHDIR)/.nojekyll
	@printf "\nThe html documentation has been collected in \"$(PUBLISHDIR)\"\n"
.PHONY: publish

# Import variables from the main Makefile
include ../Makefile

# Additional variables
SPHINXOPTS  =
SPHINXBUILD = $(python) -m sphinx
SOURCEDIR   = .
BUILDDIR    = _build
STATICDIR   = _static
PUBLISHDIR  = publish

# HTML documentation using Sphinx plus various cleanups
html: Makefile
	@# Forces Sphinx to remake HTML
	@touch index.rst
	@# Build HTML with Sphinx
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@# Change GitHub button text and link
	@for filename in $$(find $(BUILDDIR)/html/ -name '*.html'); do                      \
	    sed -i 's/Edit on GitHub/Code Repository/g' $${filename} ;                      \
	    sed -i 's/github\.com\/\/[^"]*/github\.com\/jmd-dk\/concept\//g' $${filename} ; \
	done
	@# Write CO𝘕CEPT in the heading with an italics N using HTML tags.
	@# Also insert a small (en) space after N, needed due to the heading font.
	@sed -i 's/<h1>CONCEPT/<h1>CO<em>N<\/em> CEPT/g' $(BUILDDIR)/html/index.html
	@# Generally write the N in CO𝘕CEPT in italics via HTML tags
	@for filename in $$(find $(BUILDDIR)/html/ -name '*.html'); do \
	    sed -i 's/CONCEPT/CO<em>N<\/em>CEPT/g' $${filename} ;      \
	done
	@# Remove HTML tags from titles
	@for filename in $$(find $(BUILDDIR)/html/ -name '*.html'); do                                    \
	    sed -i '/<title>.*CO<em>N<\/em>CEPT.*<\/title>/ s/CO<em>N<\/em>CEPT/CONCEPT/g' $${filename} ; \
	done
	@# Insert current date in place of |today|
	@sed -i "s/|today|/$$(date '+%b %d, %Y')/g" $(BUILDDIR)/html/index.html
	@# Remove "Next" and "Previous" buttons between chapters
	@sed -i 's/.*Previous<.*//g' $(BUILDDIR)/html/index.html
	@sed -i 's/.*>Next .*//g'    $(BUILDDIR)/html/index.html
	@sed -i 's/.*Previous<.*//g' $(BUILDDIR)/html/tutorial/tutorial.html
	@sed -i 's/.*>Next .*//g'    $(BUILDDIR)/html/tutorial/utilities.html
	@sed -i 's/.*Previous<.*//g' $(BUILDDIR)/html/installation.html
	@sed -i 's/.*>Next .*//g'    $(BUILDDIR)/html/installation.html
	@sed -i 's/.*Previous<.*//g' $(BUILDDIR)/html/command_line_options.html
	@sed -i 's/.*>Next .*//g'    $(BUILDDIR)/html/command_line_options.html
	@sed -i 's/.*Previous<.*//g' $(BUILDDIR)/html/parameters/parameters.html
	@sed -i 's/.*>Next .*//g'    $(BUILDDIR)/html/parameters/debugging_options.html
	@sed -i 's/.*Previous<.*//g' $(BUILDDIR)/html/utilities/utilities.html
	@for filename in $$(find $(BUILDDIR)/html/utilities/ -name '*.html'); do \
	    sed -i 's/.*Previous<.*//g' $${filename} ; \
	    sed -i 's/.*>Next .*//g' $${filename} ; \
	done
	@sed -i 's/.*Previous<.*//g' $(BUILDDIR)/html/under_the_hood.html
	@sed -i 's/.*>Next .*//g'    $(BUILDDIR)/html/under_the_hood.html
	@sed -i 's/.*>Next .*//g'    $(BUILDDIR)/html/troubleshooting.html
	@sed -i 's/.*Previous<.*//g' $(BUILDDIR)/html/troubleshooting.html
	@sed -i 's/.*>Next .*//g'    $(BUILDDIR)/html/publications.html
	@sed -i 's/.*Previous<.*//g' $(BUILDDIR)/html/publications.html
	@# Convert all tables in the parameter sections to plain tables
	@for filename in $$(find $(BUILDDIR)/html/parameters/ -name '*.html'); do \
	    sed -i 's/<table class="docutils.*"/<table/g' $${filename} ; \
	done



# Sphinx clean target, removing all build and publish files,
# being careful not to remove .git* files and directories.
clean_sphinx:
	$(RM) -r $(BUILDDIR) $(PUBLISHDIR)/{.[!.git],}*
.PHONY: clean_sphinx


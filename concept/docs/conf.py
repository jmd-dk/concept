# Project
project = 'CONCEPT'
copyright = '2019, Jeppe Dakin'
author = 'Jeppe Dakin'

# Paths
html_static_path = ['_static']
exclude_patterns = ['_build']

# Theme
import sphinx_rtd_theme
extensions = ['sphinx_rtd_theme']
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': 'https://jmd-dk.github.io/concept/',
    'logo_only': False,
    'display_version': False,
    'prev_next_buttons_location': 'both',
    'style_external_links': True,
    # Navigation
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': -1,
    'titles_only': False,
    'style_nav_header_background': 'darkorange',
}

# HTML
html_logo = html_static_path[0] + '/logo.png'
html_favicon = html_static_path[0] + '/favicon.ico'
html_title = 'CONCEPT Documentation'
html_context = {'display_github': True}
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = False
html_last_updated_fmt = None
html_use_smartypants = True
html_scaled_image_link = False

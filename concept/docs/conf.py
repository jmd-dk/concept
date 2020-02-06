# Project
project   = 'CONCEPT'
author    = 'Jeppe Dakin'
copyright = '2020, Jeppe Dakin'

# Paths
html_static_path = ['_static']
html_css_files = ['custom.css']
exclude_patterns = ['_build']

# Extensions
extensions = [
    'sphinx_copybutton',
    'sphinx_rtd_theme',
    'sphinx_tabs.tabs',
]

# Theme
import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    # General
    'canonical_url'             : 'https://jmd-dk.github.io/concept/',
    'display_version'           : False,
    'logo_only'                 : False,
    'prev_next_buttons_location': 'both',
    'style_external_links'      : True,
    # Navigation
    'collapse_navigation'        : False,
    'navigation_depth'           : -1,
    'sticky_navigation'          : True,
    'style_nav_header_background': 'darkorange',
    'titles_only'                : False,
}

# HTML
html_context           = {'display_github': True}
html_favicon           = html_static_path[0] + '/favicon.ico'
html_last_updated_fmt  = None
html_logo              = html_static_path[0] + '/logo.png'
html_scaled_image_link = False
html_show_copyright    = False
html_show_sourcelink   = False
html_show_sphinx       = False
html_title             = 'CONCEPT Documentation'
html_use_smartypants   = True


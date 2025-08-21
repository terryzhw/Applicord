"""Centralized GUI styling constants and utilities."""

DARK_THEME = {
    'background': '#2b2b2b',
    'text': '#ffffff',
    'button_bg': '#404040',
    'button_hover': '#505050',
    'button_pressed': '#303030',
    'button_special': '#333333',
    'button_special_hover': '#24292e',
    'border': '#555555',
    'input_bg': '#404040',
}

COMMON_STYLES = {
    'title': "font-size: 18px; font-weight: bold; margin: 20px;",
    'input_margin_small': "margin-bottom: 10px;",
    'input_margin_large': "margin-bottom: 20px;",
    'button_small': """
        QPushButton {
            font-size: 11px;
            padding: 10px;
            margin: 5px;
            text-align: center;
        }
    """,
}

MESSAGE_BOX_STYLE = f"""
    QMessageBox {{
        background-color: {DARK_THEME['background']};
        color: {DARK_THEME['text']};
    }}
    QMessageBox QPushButton {{
        background-color: {DARK_THEME['button_bg']};
        color: {DARK_THEME['text']};
        border: 1px solid {DARK_THEME['border']};
        border-radius: 5px;
        padding: 8px;
        min-width: 60px;
    }}
"""

MAIN_WINDOW_STYLE = f"""
    QMainWindow {{
        background-color: {DARK_THEME['background']};
        color: {DARK_THEME['text']};
    }}
    QWidget {{
        background-color: {DARK_THEME['background']};
        color: {DARK_THEME['text']};
    }}
    QPushButton {{
        background-color: {DARK_THEME['button_bg']};
        color: {DARK_THEME['text']};
        border: 1px solid {DARK_THEME['border']};
        border-radius: 5px;
        padding: 8px;
        font-size: 12px;
    }}
    QPushButton:hover {{
        background-color: {DARK_THEME['button_hover']};
    }}
    QPushButton:pressed {{
        background-color: {DARK_THEME['button_pressed']};
    }}
    QLineEdit {{
        background-color: {DARK_THEME['input_bg']};
        color: {DARK_THEME['text']};
        border: 1px solid {DARK_THEME['border']};
        border-radius: 5px;
        padding: 8px;
        font-size: 12px;
    }}
    QLabel {{
        color: {DARK_THEME['text']};
        font-size: 14px;
    }}
"""

def get_special_button_style():
    """Get style for special buttons like LinkedIn/GitHub."""
    return f"""
        QPushButton {{
            background-color: {DARK_THEME['button_special']};
            {COMMON_STYLES['button_small']}
        }}
        QPushButton:hover {{
            background-color: {DARK_THEME['button_special_hover']};
        }}
    """
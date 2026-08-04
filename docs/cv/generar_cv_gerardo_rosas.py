from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Gerardo_Gabriel_Rosas_Rodriguez_CV_2026.pdf"

NAVY = HexColor("#132238")
INK = HexColor("#263548")
TEAL = HexColor("#16A6A1")
MIST = HexColor("#EAF2F3")
LIGHT = HexColor("#F6F8FA")
MUTED = HexColor("#647386")
WHITE = HexColor("#FFFFFF")
LINE = HexColor("#D8E0E7")

pdfmetrics.registerFont(TTFont("Segoe", r"C:\Windows\Fonts\segoeui.ttf"))
pdfmetrics.registerFont(TTFont("Segoe-Bold", r"C:\Windows\Fonts\segoeuib.ttf"))
pdfmetrics.registerFont(TTFont("Segoe-Light", r"C:\Windows\Fonts\segoeuil.ttf"))
pdfmetrics.registerFont(TTFont("Segoe-Semibold", r"C:\Windows\Fonts\seguisb.ttf"))

STYLES = {
    "body": ParagraphStyle(
        "body",
        fontName="Segoe",
        fontSize=8.4,
        leading=11.6,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=0,
    ),
    "small": ParagraphStyle(
        "small",
        fontName="Segoe",
        fontSize=7.4,
        leading=10.2,
        textColor=MUTED,
    ),
    "sidebar": ParagraphStyle(
        "sidebar",
        fontName="Segoe",
        fontSize=7.6,
        leading=10.8,
        textColor=WHITE,
    ),
    "bullet": ParagraphStyle(
        "bullet",
        fontName="Segoe",
        fontSize=8.1,
        leading=10.9,
        textColor=INK,
        leftIndent=8,
        firstLineIndent=-8,
    ),
}


def paragraph(pdf, text, x, y, width, style="body"):
    item = Paragraph(text, STYLES[style])
    _, height = item.wrap(width, 1000)
    item.drawOn(pdf, x, y - height)
    return y - height


def heading(pdf, text, x, y, width, light=False):
    color = WHITE if light else NAVY
    pdf.setFillColor(color)
    pdf.setFont("Segoe-Semibold", 9)
    pdf.drawString(x, y, text.upper())
    pdf.setStrokeColor(TEAL)
    pdf.setLineWidth(1.5)
    pdf.line(x, y - 2.4 * mm, x + min(width, 15 * mm), y - 2.4 * mm)
    return y - 7 * mm


def bullet(pdf, text, x, y, width):
    return paragraph(pdf, f"<font color='#16A6A1'>●</font>&nbsp; {text}", x, y, width, "bullet") - 1.4 * mm


def role(pdf, title, organization, period, x, y, width):
    pdf.setFillColor(NAVY)
    pdf.setFont("Segoe-Semibold", 9)
    pdf.drawString(x, y, title)
    pdf.setFillColor(TEAL)
    pdf.setFont("Segoe", 7.8)
    pdf.drawRightString(x + width, y, period)
    pdf.setFillColor(MUTED)
    pdf.setFont("Segoe", 7.8)
    pdf.drawString(x, y - 4.1 * mm, organization)
    return y - 8.2 * mm


def link_line(pdf, label, value, url, x, y, color=WHITE):
    pdf.setFillColor(color)
    pdf.setFont("Segoe", 7.2)
    pdf.drawString(x, y, label)
    pdf.setFont("Segoe-Semibold", 7.2)
    shown = value
    pdf.drawString(x, y - 3.8 * mm, shown)
    width = pdf.stringWidth(shown, "Segoe-Semibold", 7.2)
    pdf.linkURL(url, (x, y - 4.4 * mm, x + width, y - 0.5 * mm), relative=0)
    return y - 9.3 * mm


def page_number(pdf, number):
    pdf.setFillColor(MUTED)
    pdf.setFont("Segoe", 6.8)
    pdf.drawRightString(198 * mm, 9 * mm, f"GERARDO ROSAS  ·  {number} / 2")


def draw_page_one(pdf):
    width, height = A4
    sidebar = 65 * mm

    pdf.setFillColor(NAVY)
    pdf.rect(0, 0, sidebar, height, stroke=0, fill=1)
    pdf.setFillColor(LIGHT)
    pdf.rect(sidebar, 0, width - sidebar, height, stroke=0, fill=1)

    pdf.setFillColor(TEAL)
    pdf.rect(sidebar, height - 5 * mm, width - sidebar, 5 * mm, stroke=0, fill=1)
    pdf.setFillColor(NAVY)
    pdf.setFont("Segoe-Light", 25)
    pdf.drawString(76 * mm, height - 25 * mm, "GERARDO GABRIEL")
    pdf.setFont("Segoe-Bold", 25)
    pdf.drawString(76 * mm, height - 36 * mm, "ROSAS RODRÍGUEZ")
    pdf.setFillColor(TEAL)
    pdf.setFont("Segoe-Semibold", 10)
    pdf.drawString(76 * mm, height - 45 * mm, "INGENIERO DE SOFTWARE  ·  FULL STACK  ·  IA APLICADA")

    sx, sw = 13 * mm, 42 * mm
    y = height - 18 * mm
    y = heading(pdf, "Contacto", sx, y, sw, light=True)
    y = link_line(pdf, "Ciudad de México", "756 101 9626", "tel:+527561019626", sx, y)
    y = link_line(pdf, "Correo", "bruster_30@outlook.com", "mailto:bruster_30@outlook.com", sx, y)
    y = link_line(
        pdf,
        "GitHub",
        "github.com/GerardoRosas-27",
        "https://github.com/GerardoRosas-27",
        sx,
        y,
    )
    y = link_line(
        pdf,
        "Portafolio",
        "portafolio web",
        "https://portafolio-1f3d3.firebaseapp.com/portafolio",
        sx,
        y,
    )

    y -= 2 * mm
    y = heading(pdf, "Fortalezas", sx, y, sw, light=True)
    for text in (
        "Arquitectura full stack",
        "Interfaces y sistemas web",
        "Integración de APIs y pagos",
        "Diseño de experimentos con IA",
        "Pensamiento sistémico",
        "Mentoría y enseñanza",
    ):
        y = paragraph(pdf, f"— {text}", sx, y, sw, "sidebar") - 1.1 * mm

    y -= 3 * mm
    y = heading(pdf, "Tecnologías", sx, y, sw, light=True)
    y = paragraph(
        pdf,
        "<b>Front-end</b><br/>React · Angular · Next.js · TypeScript · JavaScript · HTML · CSS<br/><br/>"
        "<b>Back-end</b><br/>Node.js · Java · Spring Boot · Python · PHP · APIs REST<br/><br/>"
        "<b>Datos y plataforma</b><br/>SQL Server · PostgreSQL · MySQL · MongoDB · Firebase · Docker · Git · CI/CD<br/><br/>"
        "<b>IA e investigación</b><br/>Rust · Candle · Gemma 2 · sistemas termodinámicos · fasores · tensor networks",
        sx,
        y,
        sw,
        "sidebar",
    )

    x, main_width = 76 * mm, 120 * mm
    y = height - 60 * mm
    y = heading(pdf, "Visión", x, y, main_width)
    pdf.setFillColor(MIST)
    pdf.roundRect(x, y - 31 * mm, main_width, 31 * mm, 3 * mm, stroke=0, fill=1)
    y = paragraph(
        pdf,
        "<b>El desarrollador no desaparece: cambia de escala.</b> Con un agente de IA de frontera, "
        "el lenguaje de programación deja de ser la barrera principal. Lo decisivo es formular el "
        "problema, construir la lógica, conectar ideas y validar evidencia. La IA amplía nuestra "
        "capacidad de ejecución; el criterio humano conserva la dirección, el contexto y la responsabilidad.",
        x + 6 * mm,
        y - 5 * mm,
        main_width - 12 * mm,
    )
    y -= 12 * mm

    y = heading(pdf, "Perfil", x, y, main_width)
    y = paragraph(
        pdf,
        "Ingeniero en Tecnologías de la Información con <b>8 años de experiencia</b> creando productos "
        "web y sistemas empresariales de extremo a extremo. Especializado en front-end, con práctica "
        "full stack en integraciones, servicios, datos y despliegue. Actualmente combino la ingeniería "
        "de producto con investigación independiente sobre arquitecturas de IA verificables y eficientes.",
        x,
        y,
        main_width,
    ) - 6 * mm

    y = heading(pdf, "Experiencia reciente", x, y, main_width)
    y = role(
        pdf,
        "Desarrollador Front-End / Full Stack",
        "EON Igniting Business",
        "DIC 2020 — ACTUALIDAD",
        x,
        y,
        main_width,
    )
    y = bullet(
        pdf,
        "<b>Círculo de Atención:</b> componentes React 18 para seguimiento de reportes; integración "
        "de pagos OpenPay/PayPal, correo y notificaciones push con Firebase Cloud Messaging.",
        x,
        y,
        main_width,
    )
    y = bullet(
        pdf,
        "Empaquetado de interfaces React como portlets personalizados para <b>Liferay DXP</b> e "
        "integración con servicios REST en Java/Spring Boot.",
        x,
        y,
        main_width,
    )
    y = bullet(
        pdf,
        "<b>Administración de Tarjetas:</b> módulos Angular 14 para gestión de SKU, comunicación "
        "con backend Python, PostgreSQL y despliegues con Docker.",
        x,
        y,
        main_width,
    )
    y = bullet(
        pdf,
        "<b>GoldenRecordAPP:</b> interfaces Angular, autenticación JWT, consumo de APIs Spring Boot "
        "y operación sobre SQL Server dentro de equipos Scrum.",
        x,
        y,
        main_width,
    )

    y -= 3 * mm
    y = heading(pdf, "Enfoque profesional", x, y, main_width)
    paragraph(
        pdf,
        "Traduzco necesidades ambiguas en flujos claros, componentes mantenibles e integraciones "
        "observables. Uso agentes de IA para explorar alternativas y acelerar la implementación, "
        "manteniendo revisión humana, pruebas y trazabilidad como parte del producto.",
        x,
        y,
        main_width,
    )
    page_number(pdf, 1)


def draw_page_two(pdf):
    width, height = A4
    pdf.setFillColor(LIGHT)
    pdf.rect(0, 0, width, height, stroke=0, fill=1)
    pdf.setFillColor(NAVY)
    pdf.rect(0, height - 29 * mm, width, 29 * mm, stroke=0, fill=1)
    pdf.setFillColor(TEAL)
    pdf.rect(0, height - 31 * mm, width, 2 * mm, stroke=0, fill=1)
    pdf.setFillColor(WHITE)
    pdf.setFont("Segoe-Light", 17)
    pdf.drawString(14 * mm, height - 18 * mm, "EXPERIENCIA, INVESTIGACIÓN Y FORMACIÓN")

    left_x, left_w = 14 * mm, 54 * mm
    right_x, right_w = 77 * mm, 119 * mm
    y_left = height - 43 * mm

    y_left = heading(pdf, "Proyecto personal", left_x, y_left, left_w)
    pdf.setFillColor(NAVY)
    pdf.roundRect(left_x, y_left - 45 * mm, left_w, 45 * mm, 3 * mm, stroke=0, fill=1)
    paragraph(
        pdf,
        "<font color='#FFFFFF'><b>CDT–RQM–EPR</b><br/>Sistema operativo cognitivo experimental en "
        "Rust. Investigación independiente sobre memoria, inferencia y consolidación para IA.</font>",
        left_x + 5 * mm,
        y_left - 6 * mm,
        left_w - 10 * mm,
        "sidebar",
    )
    pdf.setFillColor(TEAL)
    pdf.setFont("Segoe-Semibold", 7.2)
    pdf.drawString(left_x + 5 * mm, y_left - 37 * mm, "REPOSITORIO")
    repo_text = "neuro-geometrica_multi-agente"
    pdf.setFillColor(WHITE)
    pdf.setFont("Segoe", 6.8)
    pdf.drawString(left_x + 5 * mm, y_left - 41.5 * mm, repo_text)
    pdf.linkURL(
        "https://github.com/GerardoRosas-27/neuro-geometrica_multi-agente",
        (left_x + 5 * mm, y_left - 43 * mm, left_x + 49 * mm, y_left - 39 * mm),
        relative=0,
    )
    y_left -= 53 * mm

    y_left = heading(pdf, "Formación", left_x, y_left, left_w)
    y_left = paragraph(
        pdf,
        "<b>Ingeniería en Tecnologías de la Información</b><br/>Universidad de la Región Norte<br/>"
        "2014 — 2018<br/><font color='#647386'>Cédula profesional: 11236239</font>",
        left_x,
        y_left,
        left_w,
    ) - 6 * mm
    y_left = paragraph(
        pdf,
        "<b>Formación continua</b><br/>Máster en Next.js · Platzi<br/>Aprendizaje autodirigido en Rust, "
        "modelos de lenguaje y sistemas cognitivos.",
        left_x,
        y_left,
        left_w,
    )
    y_left -= 8 * mm

    y_left = heading(pdf, "Trayectoria", left_x, y_left, left_w)
    for title, period in (
        ("Branchbit · Front-End", "2020"),
        ("GTEC Software · Front-End", "2020"),
        ("Grupo Difusión Científica · Full Stack", "2019"),
        ("3e de México · Full Stack", "2018"),
    ):
        pdf.setFillColor(INK)
        pdf.setFont("Segoe-Semibold", 7.6)
        pdf.drawString(left_x, y_left, title)
        pdf.setFillColor(MUTED)
        pdf.setFont("Segoe", 7.2)
        pdf.drawRightString(left_x + left_w, y_left, period)
        y_left -= 6.5 * mm

    y = height - 43 * mm
    y = heading(pdf, "Investigación de frontera en IA", right_x, y, right_w)
    y = paragraph(
        pdf,
        "Diseño e implemento una arquitectura experimental que separa el lenguaje de la memoria y del "
        "razonamiento: <b>Gemma 2 funciona como periferia lingüística</b>, mientras motores nativos "
        "en Rust exploran, verifican y consolidan relaciones.",
        right_x,
        y,
        right_w,
    ) - 3 * mm
    y = bullet(
        pdf,
        "Motor fasorial termodinámico con descenso de energía libre, atractores y memoria de dos "
        "velocidades; consolidación transaccional con rollback.",
        right_x,
        y,
        right_w,
    )
    y = bullet(
        pdf,
        "Integración nativa de Gemma 2 cuantizado mediante Candle, con KV cache y fallback seguro "
        "cuando una ruta adaptativa no conserva la salida.",
        right_x,
        y,
        right_w,
    )
    y = bullet(
        pdf,
        "Benchmarks reproducibles y gates científicos en CI para memoria, composición, transferencia "
        "estructural, abstención fuera de distribución y descubrimiento limitado de simetrías.",
        right_x,
        y,
        right_w,
    )
    y = bullet(
        pdf,
        "Exploración de líquidos de espines y redes tensoriales —VMC, DMRG y topología pyrochlore— "
        "como sustratos computacionales medibles, sin presentar resultados experimentales como AGI.",
        right_x,
        y,
        right_w,
    )

    pdf.setFillColor(MIST)
    pdf.roundRect(right_x, y - 25 * mm, right_w, 25 * mm, 3 * mm, stroke=0, fill=1)
    paragraph(
        pdf,
        "<b>Resultado verificable del proyecto</b><br/>En un experimento interno pareado, la "
        "consolidación elevó la recuperación de patrones corrompidos de 0/144 a 144/144 en el fixture "
        "controlado y se repitió con ocho semillas. El alcance se documenta explícitamente: evidencia "
        "de deformación de cuenca, no de generalización conceptual.",
        right_x + 5 * mm,
        y - 5 * mm,
        right_w - 10 * mm,
    )
    y -= 33 * mm

    y = heading(pdf, "Experiencia anterior seleccionada", right_x, y, right_w)
    y = role(pdf, "Desarrollador Front-End", "Branchbit · PuntoVenta", "JUN — SEP 2020", right_x, y, right_w)
    y = paragraph(
        pdf,
        "Angular 8, Material UI y formularios reactivos para productos y ventas; integración con "
        "Node.js/Express, JWT y servicios Firebase.",
        right_x,
        y,
        right_w,
    ) - 5 * mm
    y = role(pdf, "Desarrollador Front-End", "GTEC Software S.A. de C.V.", "ENE — MAY 2020", right_x, y, right_w)
    y = paragraph(
        pdf,
        "Interfaces React 16 con Hooks, rutas protegidas, Material UI, pruebas con Jest/Testing "
        "Library y optimización mediante carga diferida y memoización.",
        right_x,
        y,
        right_w,
    ) - 5 * mm
    y = role(
        pdf,
        "Desarrollador Full Stack",
        "Grupo Difusión Científica · KlikDocente",
        "ENE — DIC 2019",
        right_x,
        y,
        right_w,
    )
    y = paragraph(
        pdf,
        "Sistema educativo con HTML, CSS, JavaScript, jQuery y Bootstrap; APIs Node.js/Express, "
        "autenticación JWT y persistencia con MongoDB/Firebase.",
        right_x,
        y,
        right_w,
    ) - 5 * mm
    y = role(
        pdf,
        "Desarrollador Full Stack",
        "3e de México · IMCI / Cervexxa",
        "AGO — DIC 2018",
        right_x,
        y,
        right_w,
    )
    paragraph(
        pdf,
        "Gestión de contenido y cursos con Firebase; interfaces responsivas y concepto de suscripción "
        "desplegado en GitHub Pages.",
        right_x,
        y,
        right_w,
    )
    page_number(pdf, 2)


def build():
    pdf = canvas.Canvas(str(OUTPUT), pagesize=A4)
    pdf.setTitle("CV 2026 · Gerardo Gabriel Rosas Rodríguez")
    pdf.setAuthor("Gerardo Gabriel Rosas Rodríguez")
    pdf.setSubject("Ingeniería de software, desarrollo full stack e investigación independiente en IA")
    pdf.setCreator("ReportLab · fuente versionable incluida en el repositorio")
    draw_page_one(pdf)
    pdf.showPage()
    draw_page_two(pdf)
    pdf.showPage()
    pdf.save()
    print(OUTPUT)


if __name__ == "__main__":
    build()

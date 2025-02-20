# PyQRA

PyQRA es una implementación en Python del algoritmo QR para el cálculo de autovalores y autovectores a partir de matrices como, por ejemplo, de rigidez y masa. Está diseñado para su uso educativo en la Cátedra de Álgebra Lineal Numérica de la Universidad Nacional de Rosario (UNR), pero es libre para que cualquier persona lo use, modifique y contribuya.

## Características
- **PyQRA_RUN.py**: Archivo principal donde se configuran las matrices y se ejecutan los distintos algoritmos QR.
- **QRAlgorithm_pure.py**: Implementación del algoritmo QR sin desplazamientos, útil para visualizar el proceso interno.
- **QRAlgorithm_shift.py**: Implementación del algoritmo QR con desplazamientos (Rayleigh o Wilkinson) para mejorar la convergencia.
- **/Prog/**: Próximamente incluirá casos de prueba con matrices predefinidas para evaluar ambos métodos.
- **/TESTCASE/**: [Próximamente] incluirá casos de prueba con matrices predefinidas para evaluar ambos métodos.

## Instalación y Uso
Se requiere Python 3.x y las siguientes bibliotecas:
```bash
pip install numpy
```
Para ejecutar el código:
```bash
python PyQRA_RUN.py
```

## Contribuciones
Se aceptan mejoras, comentarios y optimizaciones. Puedes abrir un issue o hacer un pull request en el repositorio.

## Licencia
Este proyecto está bajo la licencia MIT.

---
MIT License

Copyright (c) 2025 SantiagoMR

Se concede permiso, de forma gratuita, a cualquier persona que obtenga una copia de este software y los archivos de documentación asociados (el "Software"), para utilizar el Software sin restricciones, incluyendo, sin limitación, los derechos de usar, copiar, modificar, fusionar, publicar, distribuir, sublicenciar y/o vender copias del Software, y para permitir a las personas a quienes se les proporcione el Software que lo hagan, sujeto a las siguientes condiciones:

El aviso de copyright anterior y este aviso de permiso se incluirán en todas las copias o partes sustanciales del Software.

EL SOFTWARE SE PROPORCIONA "TAL CUAL", SIN GARANTÍA DE NINGÚN TIPO, EXPRESA O IMPLÍCITA, INCLUYENDO PERO NO LIMITADO A GARANTÍAS DE COMERCIABILIDAD, IDONEIDAD PARA UN PROPÓSITO PARTICULAR Y NO INFRACCIÓN. EN NINGÚN CASO LOS AUTORES O TITULARES DEL COPYRIGHT SERÁN RESPONSABLES DE NINGÚN RECLAMO, DAÑO O OTRA RESPONSABILIDAD, YA SEA EN UNA ACCIÓN DE CONTRATO, AGRAVIO O CUALQUIER OTRA FORMA, DERIVADA DE O EN CONEXIÓN CON EL SOFTWARE O EL USO U OTRO TIPO DE ACCIONES EN EL SOFTWARE.


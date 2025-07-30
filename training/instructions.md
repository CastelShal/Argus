# Training Data Structure
This guide will help you organize and prepare your image dataset for optimal performance.

## Folder Structure
- Each folder inside the `training` directory should represent **one individual**.
    - The folder name will be used as the label (name) for that person.
    - Folder names must be unique and descriptive (e.g., `john_doe`, `jane_smith`).

## Image Requirements
- **File Types:** Use standard image formats (e.g., `.jpg`, `.jpeg`, `.png`).
- **Content:** Each image should clearly show the face of the person.
- **Lighting:** Ensure good lighting to avoid shadows or overexposure.
- **Quantity:** More images per person improve recognition accuracy. Aim for at least 2–5 images per individual.
- Any files placed directly in the `training` directory (not inside a person's folder) will be ignored.

## Example
- `training/`
    - `alice/`
        - `alice1.jpg`
        - `alice2.jpg`
        - `alice3.png`
    - `bob/`
        - `bob1.jpg`
        - `bob2.jpg`

## Tips for Best Results
- Use recent, high-resolution images.
- Avoid sunglasses, hats, or anything that obscures the face.
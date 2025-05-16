extern "C" {
	#include <raylib.h>
}
#include <ros/ros.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

int main(int argc, char** argv){
	// Initialize the window
    const int screenWidth = 800;
    const int screenHeight = 600;
    InitWindow(screenWidth, screenHeight, "Tanksim Engine");

    // Set the target FPS
    SetTargetFPS(60);

    // Main game loop
    while (!WindowShouldClose()) {    // Detect window close button or ESC key
        // Start drawing
        BeginDrawing();

        ClearBackground(RAYWHITE);

        // Draw your stuff here
        DrawText("Hello, Raylib!", 190, 200, 20, LIGHTGRAY);

        // End drawing
        EndDrawing();
    }

    // De-initialize and close window
    CloseWindow();

    return 0;
}
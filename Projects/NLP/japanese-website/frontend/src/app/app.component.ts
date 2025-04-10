import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { IntelligentTextAreaComponent } from './intelligent-text-area/intelligent-text-area.component';
import { MatButtonModule } from '@angular/material/button'
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-root',
  imports: [
    RouterOutlet, 
    IntelligentTextAreaComponent, 
    MatButtonModule,
    CommonModule,
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'frontend';
}

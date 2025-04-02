import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { IntelligentTextAreaComponent } from './intelligent-text-area/intelligent-text-area.component';

@Component({
  selector: 'app-root',
  imports: [
    RouterOutlet, 
    IntelligentTextAreaComponent
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'frontend';
}

import { Component, Input } from '@angular/core';
import { environment } from '../../environments/environment.development';
import { TranslationService } from '../translation.service';
import { time } from 'node:console';

@Component({
  selector: 'app-intelligent-text-area',
  imports: [],
  templateUrl: './intelligent-text-area.component.html',
  styleUrl: './intelligent-text-area.component.css',
  standalone: true
})
export class IntelligentTextAreaComponent {

  sentence: string = ""
  translation: string = ""
  translationService: TranslationService

  timer: NodeJS.Timeout | null = null
  delay = 3000

  constructor(translationService: TranslationService){
    this.translationService = translationService
  }


  onValueChange(value: string){
    this.sentence = value
    if (this.timer){
      clearTimeout(this.timer)
    }
    this.timer = setTimeout(() => this.getTranslation(this), this.delay)
  }

  getTranslation(ex: IntelligentTextAreaComponent){
    console.log(ex.translationService)
    let translationObservable = ex.translationService.getTranslation(this.sentence)
    translationObservable.subscribe(t => {
      ex.timer = null
      if (t && typeof(t) == 'string'){
        ex.translation = t
      }
    })
  }

  


}

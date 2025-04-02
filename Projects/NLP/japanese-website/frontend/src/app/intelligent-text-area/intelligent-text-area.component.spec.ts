import { ComponentFixture, TestBed } from '@angular/core/testing';

import { IntelligentTextAreaComponent } from './intelligent-text-area.component';

describe('IntelligentTextAreaComponent', () => {
  let component: IntelligentTextAreaComponent;
  let fixture: ComponentFixture<IntelligentTextAreaComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [IntelligentTextAreaComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(IntelligentTextAreaComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

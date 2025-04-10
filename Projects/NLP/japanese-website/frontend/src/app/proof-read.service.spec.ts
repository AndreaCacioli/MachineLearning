import { TestBed } from '@angular/core/testing';

import { ProofReadService } from './proof-read.service';

describe('ProofReadService', () => {
  let service: ProofReadService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ProofReadService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});

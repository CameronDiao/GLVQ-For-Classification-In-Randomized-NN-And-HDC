printf('lendo problema %s ...\n', problema);

n_entradas= 15; n_clases= 2; n_fich= 1; fich{1}= 'crx.data'; n_patrons(1)= 690;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

n_patrons_total = sum(n_patrons); n_iter=0;

for i_fich=1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end
  for i=1:n_patrons(i_fich)
  	fprintf(2,'%5.1f%%\r', 100*n_iter++/n_patrons_total);
	for j = 1:n_entradas
	  if j==1
		val={'a','b'};
	  elseif j==4
		val={'u', 'y', 'l', 't'};
	  elseif j==5
		val={'g', 'p', 'gg'};
	  elseif j==6
		val={'c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff'};
	  elseif j==7
		val={'v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o'};
	  elseif j==9 || j==10 || j==12
		val={'t', 'f'};
	  elseif j==13
		val={'g', 'p', 's'};
	  else  % entrada continua
		t=fscanf(f,'%c',1); 
		if t=='?'
		  x(i_fich,i,j)=0;
		else
		  fseek(f,-1,SEEK_CUR); x(i_fich,i,j)=fscanf(f,'%g',1); 
		end
		fscanf(f,'%c',1); continue
	  end
	  t=''; while 1
		c = fscanf(f, '%c',1);
		if c==',' break end
		t=strcat(t,c); 
	  end
	  n=length(val); a=2/(n-1); b=(1+n)/(1-n);
	  for k=1:n
		if strcmp(t,val{k})
		  x(i_fich,i,j)=a*k+b; break
		end
	  end
	end
	t = fscanf(f,'%c',1);fscanf(f,'%c',1);  	% lectura da clase e '\n'
	if t=='+'
	  cl(i_fich,i)= 0;
	elseif t=='-'
	  cl(i_fich,i)= 1;
	else
	  error('clase %c descoñecida', t)
	end
  end
  fclose(f);
end

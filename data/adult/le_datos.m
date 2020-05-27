% adult
printf('lendo problema adult...\n');

n_entradas= 14; n_clases= 2; n_fich= 2; fich{1}= 'adult.data'; n_patrons(1)= 32561; fich{2}= 'adult.test'; n_patrons(2)= 16281;

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); cl= zeros(n_fich, n_max);

discreta = [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1];
workclass = {'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'};
education = {'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'};
marital = {'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'};
occupation = {'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'};
relationship = {'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'};
race = {'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'};
sex = {'Male', 'Female'};
country = {'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'};

n_workclass=8; n_education=16; n_marital=7; n_occupation=14; n_relationship=6; n_race=5; n_sex=2; n_country=41;

for i_fich = 1:n_fich
  f=fopen(fich{i_fich}, 'r');
  if -1==f
	error('erro en fopen abrindo %s\n', fich{i_fich});
  end

  for i=1:n_patrons(i_fich)
	fprintf(2,'%5.1f%%\r', 100*i/n_patrons(i_fich));
	for j = 1:n_entradas
	  if discreta(j)==1
		s = fscanf(f,'%s',1); fscanf(f,'%c',1);
%  		printf('%s ', s)
		if strcmp(s, '?')  % entrada ausente neste patrón
		  x(i_fich,i,j)=0;
		else
		  if j==2
			n = n_workclass; p=workclass; 
		  elseif j==4
			n = n_education; p=education; 	  
		  elseif j==6
			n = n_marital; p=marital; 	  
		  elseif j==7
			n = n_occupation; p=occupation; 	  
		  elseif j==8
			n = n_relationship; p=relationship; 	  
		  elseif j==9
			n = n_race; p=race; 	  
		  elseif j==10
			n = n_sex; p=sex; 	  
		  elseif j==14
			n = n_country; p=country;
		  end
		  a = 2/(n-1); b= (1+n)/(1-n);
		  for k=1:n
			if strcmp(s, p(k))
			  x(i_fich,i,j) = a*k + b; break
			end
		  end
		end
	  else
		x(i_fich,i,j) = fscanf(f,'%g',1); fscanf(f,'%c',1);
	  end
%  	  printf('%g ', x(i_fich,i,j))
	end
	s = fscanf(f,'%s',1);  fscanf(f,'%c',1);
	if strcmp(s, '<=50K')
	  cl(i_fich,i)=0;
	elseif strcmp(s, '>50K')
	  cl(i_fich,i)=1;
	else
	  error('clase %s descoñecida\n', s)
	end
%  	printf('\n')
%      disp(x(i_fich,i,:)); disp(cl(i_fich,i))
  end
  fclose(f);
end